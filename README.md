# AVX TO NEON

- [Overview](#Overview)
- [License](#License)
- [Requirements](#requirements)
- [Guidelines](#Guidelines)
- [Test](#Test)
- [More Information](#more-information)
- [Copyright](#copyright)

## Overview

When applications using Intel intrinsic instructions are ported from the x86 architecture to the Kunpeng architecture, the instructions need to be further developed because the Arm64 instruction names and functions are different from that of x86. As a result, huge porting workloads are caused. In this project, the frequently used AVX instructions are encapsulated as independent modules to reduce repeated development workload. The AVX instructions are replaced with related NEON SIMD instructions, while the instruction names and functions remain unchanged. Users can invoke the corresponding instructions by importing related header files into the application software. 

## License

It is licensed under the [APACHE LICENSE, VERSION 2.0](https://www.apache.org/licenses/LICENSE-2.0). For more information, see the license file.. 

## Requirements

- CPU: Kunpeng 920 

## Guidelines

In the source code directory, the source directory contains the function implementation files. The avx512intrin.h, avxintrin.h, and emmintrin.h files implement instruction translation, and the avx2neon.h file contains the header files of them. Users can execute the instructions if the application software contains avx2neon.h. 
Users need to add compilation options, for example, ARCH_CFLAGS = -march=armv8-a+fp+simd+crc, when using the header file.

## Test

This project also provides interface test cases for developers. The logic implementation code of test cases is located in the tests directory, and the input data and expected output of the test cases are in the data directory. Use the following commands to perform test cases:

```
(1) cd tests
(2) make
(3) ./test
```

After the **test** command is executed, information similar to the following is displayed on the console:

```
Running Test MM512_CASTPS128_PS512

...

Running Test MM256_SET_EPI32

AVX2NEONTest Complete: Passed 265 tests: Failed 0
```

 All the instructions provided in this project have been verified on CentOS Linux release 7.6.1810 (AltArch) and EulerOS V2.0SP8, and GCC 7.3, GCC 4.8.5, and GCC 9.2.0.

## More Information

For more information, visit

<https://www.huaweicloud.com/kunpeng/software.html>

If you have questions or comments, we encourage you to create an issue on Github.  If you wish to contact the huawei team directly, you can send email to

 [kunpengcompute@huawei.com](mailto:kunpengcompute@huawei.com).

## How to Get Code

If you wish to get source code of functions listed in supportedlist.md, you can send an email to [kunpengcompute@huawei.com](mailto:kunpengcompute@huawei.com).

## Copyright

Copyright © 2020 Huawei Corporation. All rights reserved. 
