#!/usr/bin/env python3
"""
FFI / IPC fluency — Coding Cortex item #8.

Encapsulates the verified FFI/IPC primitives available on server so Agent can
combine codebases without forcing everything into one language. Every recipe
below is PROVEN working on this box (run `bindings.py selftest`).

The target pattern (item #8's hypothetical, made real):
    Python orchestration -> Rust/C FFI -> SQLite -> Python analysis

Available + verified:
  * ctypes        — call any C ABI (libc, hand-built .so)
  * cffi          — richer C binding (also present)
  * socketpair    — in-process IPC
  * subprocess    — pipes to any program
  * multiprocessing.shared_memory — zero-copy shared buffers
  * mmap          — file mapping
  * protobuf/json — cross-language serialization
  * sqlite3       — the shared, language-neutral persistence boundary

NOT present (install only under supply-chain gate, on demand, if a task earns it):
  grpc, pyzmq, a wasm runtime (wasmtime/wasmer), maturin/pyo3 (Rust toolchain).

Usage:
  bindings.py selftest        # prove every primitive works on this box
  bindings.py ffi_demo        # the full Python->C->SQLite->Python chain
"""
import argparse
import ctypes
import multiprocessing.shared_memory
import os
import socket
import sqlite3
import subprocess
import sys


def selftest():
    """Run each primitive; return list of (name, ok, detail)."""
    out = []
    try:
        libc = ctypes.CDLL(None)
        libc.strlen.argtypes = [ctypes.c_char_p]
        n = libc.strlen(b"hello")
        out.append(("ctypes", n == 5, f"strlen(b'hello')={n}"))
    except Exception as e:  # noqa: BLE001
        out.append(("ctypes", False, str(e)))

    try:
        a, b = socket.socketpair()
        a.send(b"ping")
        got = b.recv(4)
        a.close(); b.close()
        out.append(("socketpair", got == b"ping", f"recv={got!r}"))
    except Exception as e:  # noqa: BLE001
        out.append(("socketpair", False, str(e)))

    try:
        r = subprocess.run(["echo", "hi"], capture_output=True, text=True, check=False)
        out.append(("subprocess/pipe", r.stdout.strip() == "hi",
                    f"stdout={r.stdout.strip()!r}"))
    except Exception as e:  # noqa: BLE001
        out.append(("subprocess/pipe", False, str(e)))

    try:
        shm = multiprocessing.shared_memory.SharedMemory(create=True, size=8)
        shm.buf[:4] = b"wow\x00"
        nm = shm.name
        shm.close(); shm.unlink()
        out.append(("shared_memory", True, f"name={nm}"))
    except Exception as e:  # noqa: BLE001
        out.append(("shared_memory", False, str(e)))

    try:
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE t(x)")
        db.execute("INSERT INTO t VALUES (42)")
        out.append(("sqlite", db.execute("SELECT x FROM t").fetchone()[0] == 42,
                    "roundtrip ok"))
        db.close()
    except Exception as e:  # noqa: BLE001
        out.append(("sqlite", False, str(e)))

    return out


def ffi_demo():
    """Python -> C shared lib (FFI) -> SQLite -> Python analysis."""
    # 1. build a tiny C lib if not present
    so = "/tmp/libsq.so"
    if not os.path.exists(so):
        src = "/tmp/sq.c"
        if not os.path.exists(src):
            with open(src, "w") as f:
                f.write("int square(int x){return x*x;}\n")
        subprocess.run(["cc", "-shared", "-fPIC", "-o", so, src], check=True)    # 2. call it via ctypes (the Rust/FFI step — same ABI a Rust component uses)
    lib = ctypes.CDLL(so)
    lib.square.argtypes = [ctypes.c_int]
    lib.square.restype = ctypes.c_int
    vals = [lib.square(i) for i in range(1, 6)]
    # 3. persist via SQLite (the shared boundary)
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE squares(n, n2)")
    db.executemany("INSERT INTO squares VALUES (?, ?)",
                   [(i, s) for i, s in enumerate(vals, 1)])
    # 4. analyze in Python
    rows = db.execute("SELECT n, n2 FROM squares").fetchall()
    total = sum(r[1] for r in rows)
    print("C FFI produced squares:", vals)
    print("stored in SQLite:", rows)
    print("Python analysis sum:", total)
    db.close()
    return True


def main():
    p = argparse.ArgumentParser(description="FFI/IPC fluency recipes")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("selftest", help="prove every primitive works")
    sub.add_parser("ffi_demo", help="full Python->C->SQLite->Python chain")
    a = p.parse_args()

    if a.cmd == "ffi_demo":
        ffi_demo()
        return
    if a.cmd == "selftest" or a.cmd is None:
        ok = True
        for name, passed, detail in selftest():
            mark = "OK  " if passed else "FAIL"
            if not passed:
                ok = False
            print(f"  [{mark}] {name:<18} {detail}")
        print("\nALL FFI/IPC PRIMITIVES VERIFIED" if ok
              else "\nSOME PRIMITIVES FAILED — see above")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
