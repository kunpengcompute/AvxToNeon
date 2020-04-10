/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
 * Description: avx2neon unit test main
 * Author: xuqimeng
 * Create: 2019-11-05
*/

#include <stdio.h>
#include "a2ntest.h"

int main()
{
    unsigned int i;
    int passCount = 0;
    int failCount = 0;
    for (i = UT_MM_EXTRACT_EPI32; i <= UT_MM_MALLOC; i++) {
        int flag = 0;
        const char *s = RunTest((InstructionTest)i, &flag);
        printf("Running Test %s\n", s);
        if (flag) {
            passCount++;
        } else {
            printf("**FAILURE** AVX2NEONTest %s\n", s);
            failCount++;
        }
    }
    printf("AVX2NEONTest Complete: Passed %d tests : Failed %d\n", passCount, failCount);

    return 0;
}
