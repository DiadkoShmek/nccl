#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Host-only regression; no CUDA execution or network listener.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/nccl-bootstrap-error.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT
cuda_home=${CUDA_HOME:-/usr/local/cuda}
cxx=${CXX:-c++}
make -C "$root/src" "$work_dir/include/nccl.h" BUILDDIR="$work_dir" CUDA_HOME="$cuda_home"
flags=(-std=c++17 -DNCCL_OS_LINUX -DNVTX_DISABLE -I"$work_dir/include" -I"$cuda_home/include"
       -I"$root/src/include" -I"$root/src/include/plugin" -I"$root/src/include/plugin/net"
       -ffunction-sections -fdata-sections)
for unit in misc/socket os/linux misc/crypt; do
  name=${unit##*/}
  "$cxx" "${flags[@]}" -c "$root/src/$unit.cc" -o "$work_dir/$name.o"
done
"$cxx" "${flags[@]}" "$root/repro/bootstrap-ring-error.cc"   "$work_dir/socket.o" "$work_dir/linux.o" "$work_dir/crypt.o"   -Wl,--gc-sections -pthread -o "$work_dir/bootstrap-ring-error"
result=0
for mode in closed success; do
  if ! timeout 10s "$work_dir/bootstrap-ring-error" "$mode"; then
    result=1
  fi
done
exit "$result"
