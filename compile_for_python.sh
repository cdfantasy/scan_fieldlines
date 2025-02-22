#!/bin/bash

# 编译Fortran代码生成Python模块
f2py -c -m r8akherm3 fortran_source/r8akherm3.f fortran_source/r8akherm1.f fortran_source/ibc_ck.f fortran_source/r8splinck.f --f77flags="-O3 -march=native"
f2py -c -m r8herm3  r8herm3ev.f r8zonfind.f --f77flags="-O3 -march=native"