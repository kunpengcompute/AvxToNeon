/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

 */

#ifndef AVX2NEON_H
#error Never use <immintrin.h> directly; include " avx2neon.h" instead.
#endif

#include <arm_neon.h>
#include "typedefs.h"

# define RROTATE(a,n)     (((a)<<(n))|(((a)&0xffffffff)>>(32-(n))))
# define sigma_0(x)       (RROTATE((x),25) ^ RROTATE((x),14) ^ ((x)>>3))
# define sigma_1(x)       (RROTATE((x),15) ^ RROTATE((x),13) ^ ((x)>>10))
# define Sigma_0(x)       (RROTATE((x),30) ^ RROTATE((x),19) ^ RROTATE((x),10))
# define Sigma_1(x)       (RROTATE((x),26) ^ RROTATE((x),21) ^ RROTATE((x),7))

# define Ch(x,y,z)       (((x) & (y)) ^ ((~(x)) & (z)))
# define Maj(x,y,z)      (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

FORCE_INLINE __m128i _mm_sha256rnds2_epu32(__m128i a, __m128i b, __m128i k)
{
    __m128i res;
    uint32_t A[3];
    uint32_t B[3];
    uint32_t C[3];
    uint32_t D[3];
    uint32_t E[3];
    uint32_t F[3];
    uint32_t G[3];
    uint32_t H[3];
    uint32_t K[2];

    A[0] = vgetq_lane_u32(b.vect_u32, 3);
    B[0] = vgetq_lane_u32(b.vect_u32, 2);
    C[0] = vgetq_lane_u32(a.vect_u32, 3);
    D[0] = vgetq_lane_u32(a.vect_u32, 2);
    E[0] = vgetq_lane_u32(b.vect_u32, 1);
    F[0] = vgetq_lane_u32(b.vect_u32, 0);
    G[0] = vgetq_lane_u32(a.vect_u32, 1);
    H[0] = vgetq_lane_u32(a.vect_u32, 0);

    K[0] = vgetq_lane_u32(k.vect_u32, 0);
    K[1] = vgetq_lane_u32(k.vect_u32, 1);

    for (int i = 0; i < 2; i ++) {
        uint32_t T0 = Ch(E[i], F[i], G[i]) ;
        uint32_t T1 = Sigma_1(E[i]) + K[i] + H[i];
        uint32_t T2 = Maj(A[i], B[i], C[i]);

        A[i + 1] = T0 + T1 + T2 + Sigma_0(A[i]);
        B[i + 1] = A[i];
        C[i + 1] = B[i];
        D[i + 1] = C[i];
        E[i + 1] = T0 + T1 + D[i];
        F[i + 1] = E[i];
        G[i + 1] = F[i];
        H[i + 1] = G[i];
    }

    res.vect_u32 = vsetq_lane_u32(F[2], res.vect_u32, 0);
    res.vect_u32 = vsetq_lane_u32(E[2], res.vect_u32, 1);
    res.vect_u32 = vsetq_lane_u32(B[2], res.vect_u32, 2);
    res.vect_u32 = vsetq_lane_u32(A[2], res.vect_u32, 3);

    return res;
}

FORCE_INLINE __m128i _mm_sha256msg1_epu32(__m128i a, __m128i b)
{
    __asm__ __volatile__(
        "sha256su0 %[dst].4S, %[src].4S  \n\t"
        : [dst] "+w" (a)
        : [src] "w" (b)
    );
    return a;
}

FORCE_INLINE __m128i _mm_sha256msg2_epu32(__m128i a, __m128i b)
{
    __m128i res;
    uint32_t A = vgetq_lane_u32(b.vect_u32, 2);
    uint32_t B = vgetq_lane_u32(b.vect_u32, 3);

    uint32_t C = vgetq_lane_u32(a.vect_u32, 0) + sigma_1(A);
    uint32_t D = vgetq_lane_u32(a.vect_u32, 1) + sigma_1(B);
    uint32_t E = vgetq_lane_u32(a.vect_u32, 2) + sigma_1(C);
    uint32_t F = vgetq_lane_u32(a.vect_u32, 3) + sigma_1(D);

    res.vect_u32 = vsetq_lane_u32(C, res.vect_u32, 0);
    res.vect_u32 = vsetq_lane_u32(D, res.vect_u32, 1);
    res.vect_u32 = vsetq_lane_u32(E, res.vect_u32, 2);
    res.vect_u32 = vsetq_lane_u32(F, res.vect_u32, 3);

    return res;
}
