#!/bin/sh
# Linker shim for this VM: rustup ships `rust-lld` but the box has no system
# `cc`/`ld` (no gcc/binutils). glibc dev objects ARE present (libc6-dev), so we
# drive rust-lld directly in GNU mode, supplying the C runtime startup files and
# library search paths that a `cc` driver would normally add.
#
# rustc is invoked with `-C linker-flavor=ld`, so "$@" is already an ld-style
# argument list (objects, -l libs, -L paths, -pie, -o, …). We only prepend the
# crt objects and append the library dirs (incl. .linklibs, which holds a
# libgcc_s.so symlink the gcc dev package would otherwise provide).
set -e

RUST_LLD="$HOME/.rustup/toolchains/stable-aarch64-unknown-linux-gnu/lib/rustlib/aarch64-unknown-linux-gnu/bin/rust-lld"
SYSLIB="/usr/lib/aarch64-linux-gnu"
PROJ_LIBS="$(CDPATH= cd "$(dirname "$0")/.." && pwd)/.linklibs"

exec "$RUST_LLD" -flavor gnu \
  -dynamic-linker /lib/ld-linux-aarch64.so.1 \
  --pic-executable --eh-frame-hdr \
  "$SYSLIB/Scrt1.o" "$SYSLIB/crti.o" \
  "$@" \
  "$SYSLIB/crtn.o" \
  -L"$SYSLIB" -L/lib/aarch64-linux-gnu -L"$PROJ_LIBS"
