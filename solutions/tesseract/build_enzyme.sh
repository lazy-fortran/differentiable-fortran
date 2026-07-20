#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tesseract/lib /tesseract/build
lfortran --show-llvm --fast /tesseract/enzyme/heat_step_kernel.f90 \
    > /tesseract/build/kernel.ll
clang -O3 -fPIC -S -emit-llvm /tesseract/enzyme/enzyme_wrapper.c \
    -o /tesseract/build/wrapper.ll
llvm-link -S /tesseract/build/kernel.ll /tesseract/build/wrapper.ll \
    -o /tesseract/build/linked.ll
opt -load-pass-plugin=/usr/local/lib/LLVMEnzyme-19.so -passes=enzyme \
    /tesseract/build/linked.ll -S -o /tesseract/build/enzyme.ll
clang -O3 -fPIC -shared /tesseract/build/enzyme.ll \
    -o /tesseract/lib/libdf_lfortran_enzyme.so
