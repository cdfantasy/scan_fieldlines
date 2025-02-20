#!/bin/bash

# 编译Fortran代码生成Python模块
f2py -c -m r8akherm3_simsopt r8akherm3.f r8akherm1.f ibc_ck.f r8splinck.f
f2py -c -m r8herm3_simsopt  r8herm3ev.f r8zonfind.f