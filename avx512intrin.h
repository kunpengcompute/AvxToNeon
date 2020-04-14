/*
 * Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

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
#error Never use <avx512intrin.h> directly; include " avx2neon.h" instead.
#endif


#include <arm_neon.h>

#include <math.h>
#ifdef __cplusplus
using namespace std;
#endif

#include "typedefs.h"

typedef union {
    int8x16_t vect_s8[4];
    int16x8_t vect_s16[4];
    int32x4_t vect_s32[4];
    int64x2_t vect_s64[4];
    uint8x16_t vect_u8[4];
    uint16x8_t vect_u16[4];
    uint32x4_t vect_u32[4];
    uint64x2_t vect_u64[4];
    __m256i vect_i256[2];
    __m128i vect_i128[4];
} __m512i __attribute__((aligned(64)));

typedef struct {
    float32x4_t vect_f32[4];
} __m512;

typedef struct {
    float64x2_t vect_f64[4];
} __m512d;

#define _MM_FROUND_TO_NEAREST_INT    0x00
#define _MM_FROUND_TO_NEG_INF        0x01
#define _MM_FROUND_TO_POS_INF        0x02
#define _MM_FROUND_TO_ZERO           0x03
#define _MM_FROUND_CUR_DIRECTION     0x04

#define _MM_FROUND_RAISE_EXC         0x00
#define _MM_FROUND_NO_EXC            0x08

FORCE_INLINE __m512i _mm512_div_epi8(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_i128[0] = _mm_div_epi8(a.vect_i128[0], b.vect_i128[0]);
    res_m512i.vect_i128[1] = _mm_div_epi8(a.vect_i128[1], b.vect_i128[1]);
    res_m512i.vect_i128[2] = _mm_div_epi8(a.vect_i128[2], b.vect_i128[2]);
    res_m512i.vect_i128[3] = _mm_div_epi8(a.vect_i128[3], b.vect_i128[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_div_epi16(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_i128[0] = _mm_div_epi16(a.vect_i128[0], b.vect_i128[0]);
    res_m512i.vect_i128[1] = _mm_div_epi16(a.vect_i128[1], b.vect_i128[1]);
    res_m512i.vect_i128[2] = _mm_div_epi16(a.vect_i128[2], b.vect_i128[2]);
    res_m512i.vect_i128[3] = _mm_div_epi16(a.vect_i128[3], b.vect_i128[3]);
    return res_m512i;
}


FORCE_INLINE __m512i _mm512_div_epi32(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_i256[0] = _mm256_div_epi32(a.vect_i256[0], b.vect_i256[0]);
    res_m512i.vect_i256[1] = _mm256_div_epi32(a.vect_i256[1], b.vect_i256[1]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_div_epi64(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_i256[0] = _mm256_div_epi64(a.vect_i256[0], b.vect_i256[0]);
    res_m512i.vect_i256[1] = _mm256_div_epi64(a.vect_i256[1], b.vect_i256[1]);
    return res_m512i;
}
FORCE_INLINE __m512i _mm512_div_epu8(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_i128[0] = _mm_div_epu8(a.vect_i128[0], b.vect_i128[0]);
    res_m512i.vect_i128[1] = _mm_div_epu8(a.vect_i128[1], b.vect_i128[1]);
    res_m512i.vect_i128[2] = _mm_div_epu8(a.vect_i128[2], b.vect_i128[2]);
    res_m512i.vect_i128[3] = _mm_div_epu8(a.vect_i128[3], b.vect_i128[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_div_epu16(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_i128[0] = _mm_div_epu16(a.vect_i128[0], b.vect_i128[0]);
    res_m512i.vect_i128[1] = _mm_div_epu16(a.vect_i128[1], b.vect_i128[1]);
    res_m512i.vect_i128[2] = _mm_div_epu16(a.vect_i128[2], b.vect_i128[2]);
    res_m512i.vect_i128[3] = _mm_div_epu16(a.vect_i128[3], b.vect_i128[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_div_epu32(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_i256[0] = _mm256_div_epu32(a.vect_i256[0], b.vect_i256[0]);
    res_m512i.vect_i256[1] = _mm256_div_epu32(a.vect_i256[1], b.vect_i256[1]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_div_epu64(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_i256[0] = _mm256_div_epu64(a.vect_i256[0], b.vect_i256[0]);
    res_m512i.vect_i256[1] = _mm256_div_epu64(a.vect_i256[1], b.vect_i256[1]);
    return res_m512i;
}

FORCE_INLINE __m512 _mm512_div_ps(__m512 a, __m512 b)
{
    __m512 res_m512;
    res_m512.vect_f32[0] = vdivq_f32(a.vect_f32[0], b.vect_f32[0]);
    res_m512.vect_f32[1] = vdivq_f32(a.vect_f32[1], b.vect_f32[1]);
    res_m512.vect_f32[2] = vdivq_f32(a.vect_f32[2], b.vect_f32[2]);
    res_m512.vect_f32[3] = vdivq_f32(a.vect_f32[3], b.vect_f32[3]);
    return res_m512;
}

FORCE_INLINE __m512d _mm512_div_pd(__m512d a, __m512d b)
{
    __m512d res_m512d;
    res_m512d.vect_f64[0] = vdivq_f64(a.vect_f64[0], b.vect_f64[0]);
    res_m512d.vect_f64[1] = vdivq_f64(a.vect_f64[1], b.vect_f64[1]);
    res_m512d.vect_f64[2] = vdivq_f64(a.vect_f64[2], b.vect_f64[2]);
    res_m512d.vect_f64[3] = vdivq_f64(a.vect_f64[3], b.vect_f64[3]);
    return res_m512d;
}

FORCE_INLINE __m512 _mm512_div_round_ps(__m512 a, __m512 b, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
    (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
    (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
    (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) || 
    (rounding == _MM_FROUND_CUR_DIRECTION));
    (void)rounding;
    __m512 res_m512;
    res_m512.vect_f32[0] = vdivq_f32(a.vect_f32[0], b.vect_f32[0]);
    res_m512.vect_f32[1] = vdivq_f32(a.vect_f32[1], b.vect_f32[1]);
    res_m512.vect_f32[2] = vdivq_f32(a.vect_f32[2], b.vect_f32[2]);
    res_m512.vect_f32[3] = vdivq_f32(a.vect_f32[3], b.vect_f32[3]);
    return res_m512;
}

FORCE_INLINE __m512d _mm512_div_round_pd(__m512d a, __m512d b, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
    (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
    (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
    (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) || 
    (rounding == _MM_FROUND_CUR_DIRECTION));
    (void)rounding;
    __m512d res_m512d;
    res_m512d.vect_f64[0] = vdivq_f64(a.vect_f64[0], b.vect_f64[0]);
    res_m512d.vect_f64[1] = vdivq_f64(a.vect_f64[1], b.vect_f64[1]);
    res_m512d.vect_f64[2] = vdivq_f64(a.vect_f64[2], b.vect_f64[2]);
    res_m512d.vect_f64[3] = vdivq_f64(a.vect_f64[3], b.vect_f64[3]);
    return res_m512d;
}


FORCE_INLINE __m512i _mm512_add_epi8(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s8[0] = vaddq_s8(a.vect_s8[0], b.vect_s8[0]);
    res_m512i.vect_s8[1] = vaddq_s8(a.vect_s8[1], b.vect_s8[1]);
    res_m512i.vect_s8[2] = vaddq_s8(a.vect_s8[2], b.vect_s8[2]);
    res_m512i.vect_s8[3] = vaddq_s8(a.vect_s8[3], b.vect_s8[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_add_epi16(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s16[0] = vaddq_s16(a.vect_s16[0], b.vect_s16[0]);
    res_m512i.vect_s16[1] = vaddq_s16(a.vect_s16[1], b.vect_s16[1]);
    res_m512i.vect_s16[2] = vaddq_s16(a.vect_s16[2], b.vect_s16[2]);
    res_m512i.vect_s16[3] = vaddq_s16(a.vect_s16[3], b.vect_s16[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_add_epi32(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s32[0] = vaddq_s32(a.vect_s32[0], b.vect_s32[0]);
    res_m512i.vect_s32[1] = vaddq_s32(a.vect_s32[1], b.vect_s32[1]);
    res_m512i.vect_s32[2] = vaddq_s32(a.vect_s32[2], b.vect_s32[2]);
    res_m512i.vect_s32[3] = vaddq_s32(a.vect_s32[3], b.vect_s32[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_add_epi64(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s64[0] = vaddq_s64(a.vect_s64[0], b.vect_s64[0]);
    res_m512i.vect_s64[1] = vaddq_s64(a.vect_s64[1], b.vect_s64[1]);
    res_m512i.vect_s64[2] = vaddq_s64(a.vect_s64[2], b.vect_s64[2]);
    res_m512i.vect_s64[3] = vaddq_s64(a.vect_s64[3], b.vect_s64[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_adds_epi8(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s8[0] = vqaddq_s8(a.vect_s8[0], b.vect_s8[0]);
    res_m512i.vect_s8[1] = vqaddq_s8(a.vect_s8[1], b.vect_s8[1]);
    res_m512i.vect_s8[2] = vqaddq_s8(a.vect_s8[2], b.vect_s8[2]);
    res_m512i.vect_s8[3] = vqaddq_s8(a.vect_s8[3], b.vect_s8[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_adds_epi16(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s16[0] = vqaddq_s16(a.vect_s16[0], b.vect_s16[0]);
    res_m512i.vect_s16[1] = vqaddq_s16(a.vect_s16[1], b.vect_s16[1]);
    res_m512i.vect_s16[2] = vqaddq_s16(a.vect_s16[2], b.vect_s16[2]);
    res_m512i.vect_s16[3] = vqaddq_s16(a.vect_s16[3], b.vect_s16[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_adds_epu8(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_u8[0] = vqaddq_u8(a.vect_u8[0], b.vect_u8[0]);
    res_m512i.vect_u8[1] = vqaddq_u8(a.vect_u8[1], b.vect_u8[1]);
    res_m512i.vect_u8[2] = vqaddq_u8(a.vect_u8[2], b.vect_u8[2]);
    res_m512i.vect_u8[3] = vqaddq_u8(a.vect_u8[3], b.vect_u8[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_adds_epu16(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_u16[0] = vqaddq_u16(a.vect_u16[0], b.vect_u16[0]);
    res_m512i.vect_u16[1] = vqaddq_u16(a.vect_u16[1], b.vect_u16[1]);
    res_m512i.vect_u16[2] = vqaddq_u16(a.vect_u16[2], b.vect_u16[2]);
    res_m512i.vect_u16[3] = vqaddq_u16(a.vect_u16[3], b.vect_u16[3]);
    return res_m512i;
}

FORCE_INLINE __m512 _mm512_add_ps(__m512 a, __m512 b)
{
    __m512 res_m512;
    res_m512.vect_f32[0] = vaddq_f32(a.vect_f32[0], b.vect_f32[0]);
    res_m512.vect_f32[1] = vaddq_f32(a.vect_f32[1], b.vect_f32[1]);
    res_m512.vect_f32[2] = vaddq_f32(a.vect_f32[2], b.vect_f32[2]);
    res_m512.vect_f32[3] = vaddq_f32(a.vect_f32[3], b.vect_f32[3]);
    return res_m512;
}

FORCE_INLINE __m512d _mm512_add_pd(__m512d a, __m512d b)
{
    __m512d res_m512d;
    res_m512d.vect_f64[0] = vaddq_f64(a.vect_f64[0], b.vect_f64[0]);
    res_m512d.vect_f64[1] = vaddq_f64(a.vect_f64[1], b.vect_f64[1]);
    res_m512d.vect_f64[2] = vaddq_f64(a.vect_f64[2], b.vect_f64[2]);
    res_m512d.vect_f64[3] = vaddq_f64(a.vect_f64[3], b.vect_f64[3]);
    return res_m512d;
}

FORCE_INLINE __m512 _mm512_add_round_ps (__m512 a, __m512 b, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) ||
           (rounding == (_MM_FROUND_CUR_DIRECTION)));
    (void)rounding;
    a.vect_f32[0] = vaddq_f32(a.vect_f32[0], b.vect_f32[0]);
    a.vect_f32[1] = vaddq_f32(a.vect_f32[1], b.vect_f32[1]);
    a.vect_f32[2] = vaddq_f32(a.vect_f32[2], b.vect_f32[2]);
    a.vect_f32[3] = vaddq_f32(a.vect_f32[3], b.vect_f32[3]);
    return a;
}

FORCE_INLINE __m512d _mm512_add_round_pd (__m512d a, __m512d b, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) ||
           (rounding == (_MM_FROUND_CUR_DIRECTION)));
    (void)rounding;
    a.vect_f64[0] = vaddq_f64(a.vect_f64[0], b.vect_f64[0]);
    a.vect_f64[1] = vaddq_f64(a.vect_f64[1], b.vect_f64[1]);
    a.vect_f64[2] = vaddq_f64(a.vect_f64[2], b.vect_f64[2]);
    a.vect_f64[3] = vaddq_f64(a.vect_f64[3], b.vect_f64[3]);
    return a;
}

FORCE_INLINE __m512 _mm512_addn_ps (__m512 a, __m512 b)
{
    a.vect_f32[0] = vnegq_f32(vaddq_f32(a.vect_f32[0], b.vect_f32[0]));
    a.vect_f32[1] = vnegq_f32(vaddq_f32(a.vect_f32[1], b.vect_f32[1]));
    a.vect_f32[2] = vnegq_f32(vaddq_f32(a.vect_f32[2], b.vect_f32[2]));
    a.vect_f32[3] = vnegq_f32(vaddq_f32(a.vect_f32[3], b.vect_f32[3]));
    return a;
}

FORCE_INLINE __m512d _mm512_addn_pd (__m512d a, __m512d b)
{
    a.vect_f64[0] = vnegq_f64(vaddq_f64(a.vect_f64[0], b.vect_f64[0]));
    a.vect_f64[1] = vnegq_f64(vaddq_f64(a.vect_f64[1], b.vect_f64[1]));
    a.vect_f64[2] = vnegq_f64(vaddq_f64(a.vect_f64[2], b.vect_f64[2]));
    a.vect_f64[3] = vnegq_f64(vaddq_f64(a.vect_f64[3], b.vect_f64[3]));
    return a;
}

FORCE_INLINE __m512 _mm512_addn_round_ps (__m512 a, __m512 b, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) ||
           (rounding == (_MM_FROUND_CUR_DIRECTION)));
    (void)rounding;
    a.vect_f32[0] = vnegq_f32(vaddq_f32(a.vect_f32[0], b.vect_f32[0]));
    a.vect_f32[1] = vnegq_f32(vaddq_f32(a.vect_f32[1], b.vect_f32[1]));
    a.vect_f32[2] = vnegq_f32(vaddq_f32(a.vect_f32[2], b.vect_f32[2]));
    a.vect_f32[3] = vnegq_f32(vaddq_f32(a.vect_f32[3], b.vect_f32[3]));
    return a;
}

FORCE_INLINE __m512d _mm512_addn_round_pd (__m512d a, __m512d b, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) ||
           (rounding == (_MM_FROUND_CUR_DIRECTION)));
    (void)rounding;
    a.vect_f64[0] = vnegq_f64(vaddq_f64(a.vect_f64[0], b.vect_f64[0]));
    a.vect_f64[1] = vnegq_f64(vaddq_f64(a.vect_f64[1], b.vect_f64[1]));
    a.vect_f64[2] = vnegq_f64(vaddq_f64(a.vect_f64[2], b.vect_f64[2]));
    a.vect_f64[3] = vnegq_f64(vaddq_f64(a.vect_f64[3], b.vect_f64[3]));
    return a;
}

FORCE_INLINE __m512i _mm512_addsetc_epi32 (__m512i v2, __m512i v3, __mmask16 *k2_res)
{
    __m512i res, carry;
    res.vect_u32[0] = vaddq_u32(v2.vect_u32[0], v3.vect_u32[0]);
    res.vect_u32[1] = vaddq_u32(v2.vect_u32[1], v3.vect_u32[1]);
    res.vect_u32[2] = vaddq_u32(v2.vect_u32[2], v3.vect_u32[2]);
    res.vect_u32[3] = vaddq_u32(v2.vect_u32[3], v3.vect_u32[3]);
    carry.vect_u32[0] = vcltq_u32(res.vect_u32[0], v3.vect_u32[0]);
    carry.vect_u32[1] = vcltq_u32(res.vect_u32[1], v3.vect_u32[1]);
    carry.vect_u32[2] = vcltq_u32(res.vect_u32[2], v3.vect_u32[2]);
    carry.vect_u32[3] = vcltq_u32(res.vect_u32[3], v3.vect_u32[3]);
    PICK_HB_32x16(carry, k2_res);
    return res;
}

FORCE_INLINE __m512i _mm512_addsets_epi32 (__m512i v2, __m512i v3, __mmask16 *sign)
{
    __m512i res, tmp;
    res.vect_s32[0] = vaddq_s32(v2.vect_s32[0], v3.vect_s32[0]);
    res.vect_s32[1] = vaddq_s32(v2.vect_s32[1], v3.vect_s32[1]);
    res.vect_s32[2] = vaddq_s32(v2.vect_s32[2], v3.vect_s32[2]);
    res.vect_s32[3] = vaddq_s32(v2.vect_s32[3], v3.vect_s32[3]);
    tmp.vect_u32[0] = vandq_u32(v2.vect_u32[0], v3.vect_u32[0]);
    tmp.vect_u32[1] = vandq_u32(v2.vect_u32[1], v3.vect_u32[1]);
    tmp.vect_u32[2] = vandq_u32(v2.vect_u32[2], v3.vect_u32[2]);
    tmp.vect_u32[3] = vandq_u32(v2.vect_u32[3], v3.vect_u32[3]);
    PICK_HB_32x16(tmp, sign);
    return res;
}

FORCE_INLINE __m512 _mm512_addsets_ps (__m512 v2, __m512 v3, __mmask16 *sign)
{
    __m512 res;
    __m512i tmp;
    res.vect_f32[0] = vaddq_f32(v2.vect_f32[0], v3.vect_f32[0]);
    res.vect_f32[1] = vaddq_f32(v2.vect_f32[1], v3.vect_f32[1]);
    res.vect_f32[2] = vaddq_f32(v2.vect_f32[2], v3.vect_f32[2]);
    res.vect_f32[3] = vaddq_f32(v2.vect_f32[3], v3.vect_f32[3]);
    tmp.vect_u32[0] = vandq_u32(vreinterpretq_u32_f32(v2.vect_f32[0]), vreinterpretq_u32_f32(v3.vect_f32[0]));
    tmp.vect_u32[1] = vandq_u32(vreinterpretq_u32_f32(v2.vect_f32[1]), vreinterpretq_u32_f32(v3.vect_f32[1]));
    tmp.vect_u32[2] = vandq_u32(vreinterpretq_u32_f32(v2.vect_f32[2]), vreinterpretq_u32_f32(v3.vect_f32[2]));
    tmp.vect_u32[3] = vandq_u32(vreinterpretq_u32_f32(v2.vect_f32[3]), vreinterpretq_u32_f32(v3.vect_f32[3]));
    PICK_HB_32x16(tmp, sign);
    return res;
}

FORCE_INLINE __m512 _mm512_addsets_round_ps (__m512 v2, __m512 v3, __mmask16 *sign, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) ||
           (rounding == (_MM_FROUND_CUR_DIRECTION)));
    (void)rounding;
    return _mm512_addsets_ps(v2, v3, sign);
}

FORCE_INLINE __m512i _mm512_sub_epi16(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s16[0] = vsubq_s16(a.vect_s16[0], b.vect_s16[0]);
    res_m512i.vect_s16[1] = vsubq_s16(a.vect_s16[1], b.vect_s16[1]);
    res_m512i.vect_s16[2] = vsubq_s16(a.vect_s16[2], b.vect_s16[2]);
    res_m512i.vect_s16[3] = vsubq_s16(a.vect_s16[3], b.vect_s16[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_sub_epi32(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s32[0] = vsubq_s32(a.vect_s32[0], b.vect_s32[0]);
    res_m512i.vect_s32[1] = vsubq_s32(a.vect_s32[1], b.vect_s32[1]);
    res_m512i.vect_s32[2] = vsubq_s32(a.vect_s32[2], b.vect_s32[2]);
    res_m512i.vect_s32[3] = vsubq_s32(a.vect_s32[3], b.vect_s32[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_sub_epi64(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s64[0] = vsubq_s64(a.vect_s64[0], b.vect_s64[0]);
    res_m512i.vect_s64[1] = vsubq_s64(a.vect_s64[1], b.vect_s64[1]);
    res_m512i.vect_s64[2] = vsubq_s64(a.vect_s64[2], b.vect_s64[2]);
    res_m512i.vect_s64[3] = vsubq_s64(a.vect_s64[3], b.vect_s64[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_sub_epi8(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s8[0] = vsubq_s8(a.vect_s8[0], b.vect_s8[0]);
    res_m512i.vect_s8[1] = vsubq_s8(a.vect_s8[1], b.vect_s8[1]);
    res_m512i.vect_s8[2] = vsubq_s8(a.vect_s8[2], b.vect_s8[2]);
    res_m512i.vect_s8[3] = vsubq_s8(a.vect_s8[3], b.vect_s8[3]);
    return res_m512i;
}

FORCE_INLINE __m512d _mm512_sub_pd(__m512d a, __m512d b)
{
    __m512d res_m512d;
    res_m512d.vect_f64[0] = vsubq_f64(a.vect_f64[0], b.vect_f64[0]);
    res_m512d.vect_f64[1] = vsubq_f64(a.vect_f64[1], b.vect_f64[1]);
    res_m512d.vect_f64[2] = vsubq_f64(a.vect_f64[2], b.vect_f64[2]);
    res_m512d.vect_f64[3] = vsubq_f64(a.vect_f64[3], b.vect_f64[3]);
    return res_m512d;
}

FORCE_INLINE __m512 _mm512_sub_ps(__m512 a, __m512 b)
{
    __m512 res_m512;
    res_m512.vect_f32[0] = vsubq_f32(a.vect_f32[0], b.vect_f32[0]);
    res_m512.vect_f32[1] = vsubq_f32(a.vect_f32[1], b.vect_f32[1]);
    res_m512.vect_f32[2] = vsubq_f32(a.vect_f32[2], b.vect_f32[2]);
    res_m512.vect_f32[3] = vsubq_f32(a.vect_f32[3], b.vect_f32[3]);
    return res_m512;
}

FORCE_INLINE __m512d _mm512_sub_round_pd(__m512d a, __m512d b, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) ||
           (rounding == _MM_FROUND_CUR_DIRECTION));
    (void)rounding;
    __m512d res_m512d;
    res_m512d.vect_f64[0] = vsubq_f64(a.vect_f64[0], b.vect_f64[0]);
    res_m512d.vect_f64[1] = vsubq_f64(a.vect_f64[1], b.vect_f64[1]);
    res_m512d.vect_f64[2] = vsubq_f64(a.vect_f64[2], b.vect_f64[2]);
    res_m512d.vect_f64[3] = vsubq_f64(a.vect_f64[3], b.vect_f64[3]);
    return res_m512d;
}

FORCE_INLINE __m512 _mm512_sub_round_ps(__m512 a, __m512 b, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) ||
           (rounding == _MM_FROUND_CUR_DIRECTION));
    (void)rounding;
    __m512 res_m512;
    res_m512.vect_f32[0] = vsubq_f32(a.vect_f32[0], b.vect_f32[0]);
    res_m512.vect_f32[1] = vsubq_f32(a.vect_f32[1], b.vect_f32[1]);
    res_m512.vect_f32[2] = vsubq_f32(a.vect_f32[2], b.vect_f32[2]);
    res_m512.vect_f32[3] = vsubq_f32(a.vect_f32[3], b.vect_f32[3]);
    return res_m512;
}

FORCE_INLINE __m512i _mm512_subr_epi32 (__m512i v2, __m512i v3)
{
    __m512i res_m512i;
    res_m512i.vect_s32[0] = vsubq_s32(v3.vect_s32[0], v2.vect_s32[0]);
    res_m512i.vect_s32[1] = vsubq_s32(v3.vect_s32[1], v2.vect_s32[1]);
    res_m512i.vect_s32[2] = vsubq_s32(v3.vect_s32[2], v2.vect_s32[2]);
    res_m512i.vect_s32[3] = vsubq_s32(v3.vect_s32[3], v2.vect_s32[3]);
    return res_m512i;
}

FORCE_INLINE __m512 _mm512_subr_ps (__m512 v2, __m512 v3)
{
    __m512 res_m512;
    res_m512.vect_f32[0] = vsubq_f32(v3.vect_f32[0], v2.vect_f32[0]);
    res_m512.vect_f32[1] = vsubq_f32(v3.vect_f32[1], v2.vect_f32[1]);
    res_m512.vect_f32[2] = vsubq_f32(v3.vect_f32[2], v2.vect_f32[2]);
    res_m512.vect_f32[3] = vsubq_f32(v3.vect_f32[3], v2.vect_f32[3]);
    return res_m512;
}

FORCE_INLINE __m512d _mm512_subr_pd (__m512d v2, __m512d v3)
{
    __m512d res_m512d;
    res_m512d.vect_f64[0] = vsubq_f64(v3.vect_f64[0], v2.vect_f64[0]);
    res_m512d.vect_f64[1] = vsubq_f64(v3.vect_f64[1], v2.vect_f64[1]);
    res_m512d.vect_f64[2] = vsubq_f64(v3.vect_f64[2], v2.vect_f64[2]);
    res_m512d.vect_f64[3] = vsubq_f64(v3.vect_f64[3], v2.vect_f64[3]);
    return res_m512d;
}

FORCE_INLINE __m512 _mm512_subr_round_ps (__m512 v2, __m512 v3, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) ||
           (rounding == _MM_FROUND_CUR_DIRECTION));
    (void)rounding;
    return _mm512_subr_ps(v2, v3);
}

FORCE_INLINE __m512d _mm512_subr_round_pd (__m512d v2, __m512d v3, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) ||
           (rounding == _MM_FROUND_CUR_DIRECTION));
    (void)rounding;
    return _mm512_subr_pd(v2, v3);
}

FORCE_INLINE __m512i _mm512_subsetb_epi32 (__m512i v2, __m512i v3, __mmask16 *borrow)
{
    __m512i res, carry;
    res.vect_s32[0] = vsubq_s32(v2.vect_s32[0], v3.vect_s32[0]);
    res.vect_s32[1] = vsubq_s32(v2.vect_s32[1], v3.vect_s32[1]);
    res.vect_s32[2] = vsubq_s32(v2.vect_s32[2], v3.vect_s32[2]);
    res.vect_s32[3] = vsubq_s32(v2.vect_s32[3], v3.vect_s32[3]);
    carry.vect_u32[0] = vcltq_u32(v2.vect_u32[0], v3.vect_u32[0]);
    carry.vect_u32[1] = vcltq_u32(v2.vect_u32[1], v3.vect_u32[1]);
    carry.vect_u32[2] = vcltq_u32(v2.vect_u32[2], v3.vect_u32[2]);
    carry.vect_u32[3] = vcltq_u32(v2.vect_u32[3], v3.vect_u32[3]);
    PICK_HB_32x16(carry, borrow);
    return res;
}

FORCE_INLINE __m512i _mm512_subrsetb_epi32 (__m512i v2, __m512i v3, __mmask16 *borrow)
{
    return _mm512_subsetb_epi32(v3, v2, borrow);
}

FORCE_INLINE __m512i _mm512_subs_epi16(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s16[0] = vqsubq_s16(a.vect_s16[0], b.vect_s16[0]);
    res_m512i.vect_s16[1] = vqsubq_s16(a.vect_s16[1], b.vect_s16[1]);
    res_m512i.vect_s16[2] = vqsubq_s16(a.vect_s16[2], b.vect_s16[2]);
    res_m512i.vect_s16[3] = vqsubq_s16(a.vect_s16[3], b.vect_s16[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_subs_epi8(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s8[0] = vqsubq_s8(a.vect_s8[0], b.vect_s8[0]);
    res_m512i.vect_s8[1] = vqsubq_s8(a.vect_s8[1], b.vect_s8[1]);
    res_m512i.vect_s8[2] = vqsubq_s8(a.vect_s8[2], b.vect_s8[2]);
    res_m512i.vect_s8[3] = vqsubq_s8(a.vect_s8[3], b.vect_s8[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_subs_epu16(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_u16[0] = vqsubq_u16(a.vect_u16[0], b.vect_u16[0]);
    res_m512i.vect_u16[1] = vqsubq_u16(a.vect_u16[1], b.vect_u16[1]);
    res_m512i.vect_u16[2] = vqsubq_u16(a.vect_u16[2], b.vect_u16[2]);
    res_m512i.vect_u16[3] = vqsubq_u16(a.vect_u16[3], b.vect_u16[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_subs_epu8(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_u8[0] = vqsubq_u8(a.vect_u8[0], b.vect_u8[0]);
    res_m512i.vect_u8[1] = vqsubq_u8(a.vect_u8[1], b.vect_u8[1]);
    res_m512i.vect_u8[2] = vqsubq_u8(a.vect_u8[2], b.vect_u8[2]);
    res_m512i.vect_u8[3] = vqsubq_u8(a.vect_u8[3], b.vect_u8[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_permutexvar_epi32 (__m512i idx, __m512i a)
{
    __m512i res;
    int32x4_t low4bit = vdupq_n_s32(0x0f);
    idx.vect_s32[0] = vandq_s32(idx.vect_s32[0], low4bit);
    idx.vect_s32[1] = vandq_s32(idx.vect_s32[1], low4bit);
    idx.vect_s32[2] = vandq_s32(idx.vect_s32[2], low4bit);
    idx.vect_s32[3] = vandq_s32(idx.vect_s32[3], low4bit);
    int32_t p_a[16], p_i[16];
    vst1q_s32(p_a, a.vect_s32[0]);
    vst1q_s32(p_a + 4, a.vect_s32[1]);
    vst1q_s32(p_a + 8, a.vect_s32[2]);
    vst1q_s32(p_a + 12, a.vect_s32[3]);
    vst1q_s32(p_i, idx.vect_s32[0]);
    vst1q_s32(p_i + 4, idx.vect_s32[1]);
    vst1q_s32(p_i + 8, idx.vect_s32[2]);
    vst1q_s32(p_i + 12, idx.vect_s32[3]);
    res.vect_s32[0] = vsetq_lane_s32(p_a[p_i[0]], res.vect_s32[0], 0);
    res.vect_s32[0] = vsetq_lane_s32(p_a[p_i[1]], res.vect_s32[0], 1);
    res.vect_s32[0] = vsetq_lane_s32(p_a[p_i[2]], res.vect_s32[0], 2);
    res.vect_s32[0] = vsetq_lane_s32(p_a[p_i[3]], res.vect_s32[0], 3);
    res.vect_s32[1] = vsetq_lane_s32(p_a[p_i[4]], res.vect_s32[1], 0);
    res.vect_s32[1] = vsetq_lane_s32(p_a[p_i[5]], res.vect_s32[1], 1);
    res.vect_s32[1] = vsetq_lane_s32(p_a[p_i[6]], res.vect_s32[1], 2);
    res.vect_s32[1] = vsetq_lane_s32(p_a[p_i[7]], res.vect_s32[1], 3);
    res.vect_s32[2] = vsetq_lane_s32(p_a[p_i[8]], res.vect_s32[2], 0);
    res.vect_s32[2] = vsetq_lane_s32(p_a[p_i[9]], res.vect_s32[2], 1);
    res.vect_s32[2] = vsetq_lane_s32(p_a[p_i[10]], res.vect_s32[2], 2);
    res.vect_s32[2] = vsetq_lane_s32(p_a[p_i[11]], res.vect_s32[2], 3);
    res.vect_s32[3] = vsetq_lane_s32(p_a[p_i[12]], res.vect_s32[3], 0);
    res.vect_s32[3] = vsetq_lane_s32(p_a[p_i[13]], res.vect_s32[3], 1);
    res.vect_s32[3] = vsetq_lane_s32(p_a[p_i[14]], res.vect_s32[3], 2);
    res.vect_s32[3] = vsetq_lane_s32(p_a[p_i[15]], res.vect_s32[3], 3);
    return res;
}

FORCE_INLINE __m512i _mm512_permutexvar_epi64(__m512i idx, __m512i a)
{
    __m512i res;
    int64x2_t low3bit = vdupq_n_s64(0x07);
    idx.vect_s64[0] = vandq_s64(idx.vect_s64[0], low3bit);
    idx.vect_s64[1] = vandq_s64(idx.vect_s64[1], low3bit);
    idx.vect_s64[2] = vandq_s64(idx.vect_s64[2], low3bit);
    idx.vect_s64[3] = vandq_s64(idx.vect_s64[3], low3bit);
    int64_t p_a[8], p_i[8];
    vst1q_s64(p_a, a.vect_s64[0]);
    vst1q_s64(p_a + 2, a.vect_s64[1]);
    vst1q_s64(p_a + 4, a.vect_s64[2]);
    vst1q_s64(p_a + 6, a.vect_s64[3]);
    vst1q_s64(p_i, idx.vect_s64[0]);
    vst1q_s64(p_i + 2, idx.vect_s64[1]);
    vst1q_s64(p_i + 4, idx.vect_s64[2]);
    vst1q_s64(p_i + 6, idx.vect_s64[3]);
    res.vect_s64[0] = vsetq_lane_s64(p_a[p_i[0]], res.vect_s64[0], 0);
    res.vect_s64[0] = vsetq_lane_s64(p_a[p_i[1]], res.vect_s64[0], 1);
    res.vect_s64[1] = vsetq_lane_s64(p_a[p_i[2]], res.vect_s64[1], 0);
    res.vect_s64[1] = vsetq_lane_s64(p_a[p_i[3]], res.vect_s64[1], 1);
    res.vect_s64[2] = vsetq_lane_s64(p_a[p_i[4]], res.vect_s64[2], 0);
    res.vect_s64[2] = vsetq_lane_s64(p_a[p_i[5]], res.vect_s64[2], 1);
    res.vect_s64[3] = vsetq_lane_s64(p_a[p_i[6]], res.vect_s64[3], 0);
    res.vect_s64[3] = vsetq_lane_s64(p_a[p_i[7]], res.vect_s64[3], 1);
    return res;
}

FORCE_INLINE __m512i _mm512_permutex2var_epi32 (__m512i a, __m512i idx, __m512i b)
{
    __m512i res;
    int32_t *ptr_a = (int32_t *)&a;
    int32_t *ptr_b = (int32_t *)&b;
    int32_t *ptr_i = (int32_t *)&idx;
    int32_t *ptr_r = (int32_t *)&res;
    int i;
    for (i = 0; i < 16; ++i) {
        int id = ptr_i[i] & 0x0f;
        ptr_r[i] = ((ptr_i[i] & 0x10)) ? ptr_b[id] : ptr_a[id];
    }
    return res;
}

FORCE_INLINE __mmask64 _mm512_test_epi8_mask(__m512i a, __m512i b)
{
    uint8x16_t mask_and = vld1q_u8(g_mask_epi8);
    __m512i tmp;
    tmp.vect_u8[0] = vandq_u8(vtstq_u8(a.vect_u8[0], b.vect_u8[0]), mask_and);
    tmp.vect_u8[1] = vandq_u8(vtstq_u8(a.vect_u8[1], b.vect_u8[1]), mask_and);
    tmp.vect_u8[2] = vandq_u8(vtstq_u8(a.vect_u8[2], b.vect_u8[2]), mask_and);
    tmp.vect_u8[3] = vandq_u8(vtstq_u8(a.vect_u8[3], b.vect_u8[3]), mask_and);
    uint8_t r[8];
    __asm__ __volatile__ (
        "addv %b[r0], %[t0].8b              \n\t"
        "addv %b[r2], %[t1].8b              \n\t"
        "addv %b[r4], %[t2].8b              \n\t"
        "addv %b[r6], %[t3].8b              \n\t"
        "ins %[t0].d[0], %[t0].d[1]         \n\t"
        "ins %[t1].d[0], %[t1].d[1]         \n\t"
        "ins %[t2].d[0], %[t2].d[1]         \n\t"
        "ins %[t3].d[0], %[t3].d[1]         \n\t"
        "addv %b[r1], %[t0].8b              \n\t"
        "addv %b[r3], %[t1].8b              \n\t"
        "addv %b[r5], %[t2].8b              \n\t"
        "addv %b[r7], %[t3].8b              \n\t"
        :[r0]"=w"(r[0]), [r1]"=w"(r[1]), [r2]"=w"(r[2]), [r3]"=w"(r[3]), [r4]"=w"(r[4]), [r5]"=w"(r[5]), [r6]"=w"(r[6]),
         [r7]"=w"(r[7]), 
         [t0]"+w"(tmp.vect_u8[0]), [t1]"+w"(tmp.vect_u8[1]), [t2]"+w"(tmp.vect_u8[2]), [t3]"+w"(tmp.vect_u8[3])
    );
    uint64x1_t res = vreinterpret_u64_u8(vld1_u8((const uint8_t *)r));
    return vget_lane_u64(res, 0);
}

FORCE_INLINE __mmask16 _mm512_test_epi32_mask(__m512i a, __m512i b)
{
    uint32x4_t mask_and = vld1q_u32(g_mask_epi32);
    __m512i tmp;
    tmp.vect_u32[0] = vandq_u32(vtstq_u32(a.vect_u32[0], b.vect_u32[0]), mask_and);
    tmp.vect_u32[1] = vandq_u32(vtstq_u32(a.vect_u32[1], b.vect_u32[1]), mask_and);
    tmp.vect_u32[2] = vandq_u32(vtstq_u32(a.vect_u32[2], b.vect_u32[2]), mask_and);
    tmp.vect_u32[3] = vandq_u32(vtstq_u32(a.vect_u32[3], b.vect_u32[3]), mask_and);
    uint32_t r0 = vaddvq_u32(tmp.vect_u32[0]);
    uint32_t r1 = vaddvq_u32(tmp.vect_u32[1]);
    uint32_t r2 = vaddvq_u32(tmp.vect_u32[2]);
    uint32_t r3 = vaddvq_u32(tmp.vect_u32[3]);
    __mmask16 res = r0 | (r1 << 4) | (r2 << 8) | (r3 << 12);
    return res;
}

FORCE_INLINE __mmask8 _mm512_test_epi64_mask(__m512i a, __m512i b)
{
    uint64x2_t mask_and = vld1q_u64(g_mask_epi64);
    __m512i tmp;
    tmp.vect_u64[0] = vandq_u64(vtstq_u64(a.vect_u64[0], b.vect_u64[0]), mask_and);
    tmp.vect_u64[1] = vandq_u64(vtstq_u64(a.vect_u64[1], b.vect_u64[1]), mask_and);
    tmp.vect_u64[2] = vandq_u64(vtstq_u64(a.vect_u64[2], b.vect_u64[2]), mask_and);
    tmp.vect_u64[3] = vandq_u64(vtstq_u64(a.vect_u64[3], b.vect_u64[3]), mask_and);
    uint32_t r0 = vaddvq_u32(tmp.vect_u32[0]);
    uint32_t r1 = vaddvq_u32(tmp.vect_u32[1]);
    uint32_t r2 = vaddvq_u32(tmp.vect_u32[2]);
    uint32_t r3 = vaddvq_u32(tmp.vect_u32[3]);
    __mmask8 res = r0 | (r1 << 2) | (r2 << 4) | (r3 << 6);
    return res;
}

FORCE_INLINE __m512i _mm512_mul_epi32(__m512i a, __m512i b)
{
    __asm__ __volatile__ (
        "ins %[a0].s[1], %[a0].s[2]             \n\t"
        "ins %[a1].s[1], %[a1].s[2]             \n\t"
        "ins %[a2].s[1], %[a2].s[2]             \n\t"
        "ins %[a3].s[1], %[a3].s[2]             \n\t"
        "ins %[b0].s[1], %[b0].s[2]             \n\t"
        "ins %[b1].s[1], %[b1].s[2]             \n\t"
        "ins %[b2].s[1], %[b2].s[2]             \n\t"
        "ins %[b3].s[1], %[b3].s[2]             \n\t"
        "smull %[a0].2d, %[a0].2s, %[b0].2s     \n\t"
        "smull %[a1].2d, %[a1].2s, %[b1].2s     \n\t"
        "smull %[a2].2d, %[a2].2s, %[b2].2s     \n\t"
        "smull %[a3].2d, %[a3].2s, %[b3].2s     \n\t"
        :[a0]"+w"(a.vect_s32[0]), [a1]"+w"(a.vect_s32[1]), [a2]"+w"(a.vect_s32[2]), [a3]"+w"(a.vect_s32[3]), 
         [b0]"+w"(b.vect_s32[0]), [b1]"+w"(b.vect_s32[1]), [b2]"+w"(b.vect_s32[2]), [b3]"+w"(b.vect_s32[3])
        :
        :
    );
    return a;
}

FORCE_INLINE __m512i _mm512_mul_epu32(__m512i a, __m512i b)
{
    __asm__ __volatile__ (
        "ins %[a0].s[1], %[a0].s[2]             \n\t"
        "ins %[a1].s[1], %[a1].s[2]             \n\t"
        "ins %[a2].s[1], %[a2].s[2]             \n\t"
        "ins %[a3].s[1], %[a3].s[2]             \n\t"
        "ins %[b0].s[1], %[b0].s[2]             \n\t"
        "ins %[b1].s[1], %[b1].s[2]             \n\t"
        "ins %[b2].s[1], %[b2].s[2]             \n\t"
        "ins %[b3].s[1], %[b3].s[2]             \n\t"
        "umull %[a0].2d, %[a0].2s, %[b0].2s     \n\t"
        "umull %[a1].2d, %[a1].2s, %[b1].2s     \n\t"
        "umull %[a2].2d, %[a2].2s, %[b2].2s     \n\t"
        "umull %[a3].2d, %[a3].2s, %[b3].2s     \n\t"
        :[a0]"+w"(a.vect_u32[0]), [a1]"+w"(a.vect_u32[1]), [a2]"+w"(a.vect_u32[2]), [a3]"+w"(a.vect_u32[3]), 
         [b0]"+w"(b.vect_u32[0]), [b1]"+w"(b.vect_u32[1]), [b2]"+w"(b.vect_u32[2]), [b3]"+w"(b.vect_u32[3])
        :
        :
    );
    return a;
}

FORCE_INLINE __m512d _mm512_mul_pd(__m512d a, __m512d b)
{
    __m512d res_m512d;
    res_m512d.vect_f64[0] = vmulq_f64(a.vect_f64[0], b.vect_f64[0]);
    res_m512d.vect_f64[1] = vmulq_f64(a.vect_f64[1], b.vect_f64[1]);
    res_m512d.vect_f64[2] = vmulq_f64(a.vect_f64[2], b.vect_f64[2]);
    res_m512d.vect_f64[3] = vmulq_f64(a.vect_f64[3], b.vect_f64[3]);
    return res_m512d;
}

FORCE_INLINE __m512 _mm512_mul_ps(__m512 a, __m512 b)
{
    __m512 res_m512;
    res_m512.vect_f32[0] = vmulq_f32(a.vect_f32[0], b.vect_f32[0]);
    res_m512.vect_f32[1] = vmulq_f32(a.vect_f32[1], b.vect_f32[1]);
    res_m512.vect_f32[2] = vmulq_f32(a.vect_f32[2], b.vect_f32[2]);
    res_m512.vect_f32[3] = vmulq_f32(a.vect_f32[3], b.vect_f32[3]);
    return res_m512;
}

FORCE_INLINE __m512i _mm512_mulhi_epi16(__m512i a, __m512i b)
{
    __m512i res;
    res.vect_i256[0] = _mm256_mulhi_epi16(a.vect_i256[0], b.vect_i256[0]);
    res.vect_i256[1] = _mm256_mulhi_epi16(a.vect_i256[1], b.vect_i256[1]);
    return res;
}

FORCE_INLINE __m512i _mm512_mulhi_epu16(__m512i a, __m512i b)
{
    __m512i res;
    res.vect_i256[0] = _mm256_mulhi_epu16(a.vect_i256[0], b.vect_i256[0]);
    res.vect_i256[1] = _mm256_mulhi_epu16(a.vect_i256[1], b.vect_i256[1]);
    return res;
}

FORCE_INLINE __m512i _mm512_mulhi_epi32 (__m512i a, __m512i b)
{
    __m512i res;
    res.vect_i256[0] = _mm256_mulhi_epi32(a.vect_i256[0], b.vect_i256[0]);
    res.vect_i256[1] = _mm256_mulhi_epi32(a.vect_i256[1], b.vect_i256[1]);
    return res;
}

FORCE_INLINE __m512i _mm512_mulhi_epu32 (__m512i a, __m512i b)
{
    __m512i res;
    res.vect_i256[0] = _mm256_mulhi_epu32(a.vect_i256[0], b.vect_i256[0]);
    res.vect_i256[1] = _mm256_mulhi_epu32(a.vect_i256[1], b.vect_i256[1]);
    return res;
}

FORCE_INLINE __m512i _mm512_mullo_epi16(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s16[0] = vmulq_s16(a.vect_s16[0], b.vect_s16[0]);
    res_m512i.vect_s16[1] = vmulq_s16(a.vect_s16[1], b.vect_s16[1]);
    res_m512i.vect_s16[2] = vmulq_s16(a.vect_s16[2], b.vect_s16[2]);
    res_m512i.vect_s16[3] = vmulq_s16(a.vect_s16[3], b.vect_s16[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_mullo_epi32(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s32[0] = vmulq_s32(a.vect_s32[0], b.vect_s32[0]);
    res_m512i.vect_s32[1] = vmulq_s32(a.vect_s32[1], b.vect_s32[1]);
    res_m512i.vect_s32[2] = vmulq_s32(a.vect_s32[2], b.vect_s32[2]);
    res_m512i.vect_s32[3] = vmulq_s32(a.vect_s32[3], b.vect_s32[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_mullo_epi64(__m512i a, __m512i b)
{
    __m512i res;
    res.vect_i256[0] = _mm256_mullo_epi64(a.vect_i256[0], b.vect_i256[0]);
    res.vect_i256[1] = _mm256_mullo_epi64(a.vect_i256[1], b.vect_i256[1]);
    return res;
}

FORCE_INLINE __m512i _mm512_mullox_epi64(__m512i a, __m512i b)
{
    __m512i res;
    res.vect_i256[0] = _mm256_mullo_epi64(a.vect_i256[0], b.vect_i256[0]);
    res.vect_i256[1] = _mm256_mullo_epi64(a.vect_i256[1], b.vect_i256[1]);
    return res;
}

FORCE_INLINE __m512i _mm512_mulhrs_epi16(__m512i a, __m512i b)
{
    __m512i res;
    res.vect_i256[0] = _mm256_mulhrs_epi16(a.vect_i256[0], b.vect_i256[0]);
    res.vect_i256[1] = _mm256_mulhrs_epi16(a.vect_i256[1], b.vect_i256[1]);
    return res;
}

FORCE_INLINE __m512d _mm512_mul_round_pd(__m512d a, __m512d b, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) ||
           (rounding == _MM_FROUND_CUR_DIRECTION));
    (void)rounding;
    __m512d res_m512d;
    res_m512d.vect_f64[0] = vmulq_f64(a.vect_f64[0], b.vect_f64[0]);
    res_m512d.vect_f64[1] = vmulq_f64(a.vect_f64[1], b.vect_f64[1]);
    res_m512d.vect_f64[2] = vmulq_f64(a.vect_f64[2], b.vect_f64[2]);
    res_m512d.vect_f64[3] = vmulq_f64(a.vect_f64[3], b.vect_f64[3]);
    return res_m512d;
}

FORCE_INLINE __m512 _mm512_mul_round_ps(__m512 a, __m512 b, int rounding)
{
    assert((rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)) || \
           (rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)) ||
           (rounding == _MM_FROUND_CUR_DIRECTION));
    (void)rounding;
    __m512 res_m512;
    res_m512.vect_f32[0] = vmulq_f32(a.vect_f32[0], b.vect_f32[0]);
    res_m512.vect_f32[1] = vmulq_f32(a.vect_f32[1], b.vect_f32[1]);
    res_m512.vect_f32[2] = vmulq_f32(a.vect_f32[2], b.vect_f32[2]);
    res_m512.vect_f32[3] = vmulq_f32(a.vect_f32[3], b.vect_f32[3]);
    return res_m512;
}

FORCE_INLINE __m512i _mm512_sll_epi64(__m512i a, __m128i count)
{
    int c = count.vect_s64[0];
    __m512i result_m512i;
    if (likely(c >= 0 && c < 64)) {
        result_m512i.vect_s64[0] = vshlq_n_s64(a.vect_s64[0], c);
        result_m512i.vect_s64[1] = vshlq_n_s64(a.vect_s64[1], c);
        result_m512i.vect_s64[2] = vshlq_n_s64(a.vect_s64[2], c);
        result_m512i.vect_s64[3] = vshlq_n_s64(a.vect_s64[3], c);
    } else {
        result_m512i.vect_s64[0] = vdupq_n_s64(0);
        result_m512i.vect_s64[1] = vdupq_n_s64(0);
        result_m512i.vect_s64[2] = vdupq_n_s64(0);
        result_m512i.vect_s64[3] = vdupq_n_s64(0);
    } 
    return result_m512i;
}

FORCE_INLINE __m512i _mm512_slli_epi64(__m512i a, unsigned int imm8)
{
    __m512i result_m512i;
    if (likely(imm8 < 64)) {
        result_m512i.vect_s64[0] = vshlq_n_s64(a.vect_s64[0], imm8);
        result_m512i.vect_s64[1] = vshlq_n_s64(a.vect_s64[1], imm8);
        result_m512i.vect_s64[2] = vshlq_n_s64(a.vect_s64[2], imm8);
        result_m512i.vect_s64[3] = vshlq_n_s64(a.vect_s64[3], imm8);
    } else {
        result_m512i.vect_s64[0] = vdupq_n_s64(0);
        result_m512i.vect_s64[1] = vdupq_n_s64(0);
        result_m512i.vect_s64[2] = vdupq_n_s64(0);
        result_m512i.vect_s64[3] = vdupq_n_s64(0);
    }
    return result_m512i;
}

FORCE_INLINE __m512i _mm512_srli_epi64(__m512i a, unsigned int imm8)
{
    __m512i result_m512i;
    if (likely(imm8 < 64)) {
        int64x2_t vect_imm = vdupq_n_s64(-imm8);
        result_m512i.vect_u64[0] = vshlq_u64(a.vect_u64[0], vect_imm);
        result_m512i.vect_u64[1] = vshlq_u64(a.vect_u64[1], vect_imm);
        result_m512i.vect_u64[2] = vshlq_u64(a.vect_u64[2], vect_imm);
        result_m512i.vect_u64[3] = vshlq_u64(a.vect_u64[3], vect_imm);
    } else {
        result_m512i.vect_u64[0] = vdupq_n_u64(0);
        result_m512i.vect_u64[1] = vdupq_n_u64(0);
        result_m512i.vect_u64[2] = vdupq_n_u64(0);
        result_m512i.vect_u64[3] = vdupq_n_u64(0);
    } 
    return result_m512i;
}

FORCE_INLINE __m512i _mm512_bslli_epi128(__m512i a, const int imm8)
{
    assert(imm8 >= 0 && imm8 <= 255);
    __m512i res_m512i;
    if (likely(imm8 > 0 && imm8 <= 15)) {
        res_m512i.vect_s8[0] = vextq_s8(vdupq_n_s8(0), a.vect_s8[0], 16 - imm8);
        res_m512i.vect_s8[1] = vextq_s8(vdupq_n_s8(0), a.vect_s8[1], 16 - imm8);
        res_m512i.vect_s8[2] = vextq_s8(vdupq_n_s8(0), a.vect_s8[2], 16 - imm8);
        res_m512i.vect_s8[3] = vextq_s8(vdupq_n_s8(0), a.vect_s8[3], 16 - imm8);
    } else if (imm8 == 0) {
        res_m512i = a;
    } else {
        res_m512i.vect_s8[0] = vdupq_n_s8(0);
        res_m512i.vect_s8[1] = vdupq_n_s8(0);
        res_m512i.vect_s8[2] = vdupq_n_s8(0);
        res_m512i.vect_s8[3] = vdupq_n_s8(0);
    }
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_bsrli_epi128(__m512i a, const int imm8)
{
    assert(imm8 >= 0 && imm8 <= 255);
    __m512i res_m512i;
    if (likely(imm8 > 0 && imm8 <= 15)) {
        res_m512i.vect_s8[0] = vextq_s8(a.vect_s8[0], vdupq_n_s8(0), imm8);
        res_m512i.vect_s8[1] = vextq_s8(a.vect_s8[1], vdupq_n_s8(0), imm8);
        res_m512i.vect_s8[2] = vextq_s8(a.vect_s8[2], vdupq_n_s8(0), imm8);
        res_m512i.vect_s8[3] = vextq_s8(a.vect_s8[3], vdupq_n_s8(0), imm8);
    } else if (imm8 == 0) {
        res_m512i = a;
    } else {
        res_m512i.vect_s8[0] = vdupq_n_s8(0);
        res_m512i.vect_s8[1] = vdupq_n_s8(0);
        res_m512i.vect_s8[2] = vdupq_n_s8(0);
        res_m512i.vect_s8[3] = vdupq_n_s8(0);
    }
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_unpackhi_epi8(__m512i a, __m512i b)
{
    __m512i result_m512i;
    result_m512i.vect_s8[0] = vzip2q_s8(a.vect_s8[0], b.vect_s8[0]);
    result_m512i.vect_s8[1] = vzip2q_s8(a.vect_s8[1], b.vect_s8[1]);
    result_m512i.vect_s8[2] = vzip2q_s8(a.vect_s8[2], b.vect_s8[2]);
    result_m512i.vect_s8[3] = vzip2q_s8(a.vect_s8[3], b.vect_s8[3]);
    return result_m512i;
}

FORCE_INLINE __m512i _mm512_unpacklo_epi8(__m512i a, __m512i b)
{
    __m512i result_m512i;
    result_m512i.vect_s8[0] = vzip1q_s8(a.vect_s8[0], b.vect_s8[0]);
    result_m512i.vect_s8[1] = vzip1q_s8(a.vect_s8[1], b.vect_s8[1]);
    result_m512i.vect_s8[2] = vzip1q_s8(a.vect_s8[2], b.vect_s8[2]);
    result_m512i.vect_s8[3] = vzip1q_s8(a.vect_s8[3], b.vect_s8[3]);
    return result_m512i;
}

FORCE_INLINE __m512d _mm512_cmp_pd(__m512d a, __m512d b, const int imm8)
{
    assert(imm8 < 32 && imm8 >= 0);
    __m512d dst;
    dst.vect_f64[0] = (float64x2_t)g_FunListCmp256Pd[imm8].cmpFun(a.vect_f64[0], b.vect_f64[0]);
    dst.vect_f64[1] = (float64x2_t)g_FunListCmp256Pd[imm8].cmpFun(a.vect_f64[1], b.vect_f64[1]);
    dst.vect_f64[2] = (float64x2_t)g_FunListCmp256Pd[imm8].cmpFun(a.vect_f64[2], b.vect_f64[2]);
    dst.vect_f64[3] = (float64x2_t)g_FunListCmp256Pd[imm8].cmpFun(a.vect_f64[3], b.vect_f64[3]);
    return dst;
}

FORCE_INLINE __mmask8 _mm512_cmp_pd_mask(__m512d a, __m512d b, const int imm8)
{
    assert(imm8 < 32 && imm8 >= 0);
    __m512d dst = _mm512_cmp_pd(a, b, imm8);
    __mmask8 res = 0;
    uint64x2_t vect_mask = vld1q_u64(g_mask_epi64);
    __m512i tmp;
    uint64_t r[4];
    __asm__ __volatile__ (
        "and %[t0].16b, %[d0].16b, %[mask].16b        \n\t"
        "and %[t1].16b, %[d1].16b, %[mask].16b        \n\t"
        "and %[t2].16b, %[d2].16b, %[mask].16b        \n\t"
        "and %[t3].16b, %[d3].16b, %[mask].16b        \n\t"
        "addp %d[r0], %[t0].2d                        \n\t"
        "addp %d[r1], %[t1].2d                        \n\t"
        "addp %d[r2], %[t2].2d                        \n\t"
        "addp %d[r3], %[t3].2d                        \n\t"
        :[t0]"+w"(tmp.vect_u64[0]), [t1]"+w"(tmp.vect_u64[1]), [t2]"+w"(tmp.vect_u64[2]), [t3]"+w"(tmp.vect_u64[3]), 
         [r0]"=w"(r[0]), [r1]"=w"(r[1]), [r2]"=w"(r[2]), [r3]"=w"(r[3])
        :[d0]"w"(dst.vect_f64[0]), [d1]"w"(dst.vect_f64[1]), [d2]"w"(dst.vect_f64[2]), [d3]"w"(dst.vect_f64[3]), 
         [mask]"w"(vect_mask)
    );
    res = r[0] | (r[1] << 2) | (r[2] << 4) | (r[3] << 6);
    return res;
}

FORCE_INLINE __m512 _mm512_cmp_ps(__m512 a, __m512 b, const int imm8)
{
    assert(imm8 < 32 && imm8 >= 0);
    __m512 dst;
    dst.vect_f32[0] = vreinterpretq_f32_u32(g_FunListCmp256Ps[imm8].cmpFun(a.vect_f32[0], b.vect_f32[0]));
    dst.vect_f32[1] = vreinterpretq_f32_u32(g_FunListCmp256Ps[imm8].cmpFun(a.vect_f32[1], b.vect_f32[1]));
    dst.vect_f32[2] = vreinterpretq_f32_u32(g_FunListCmp256Ps[imm8].cmpFun(a.vect_f32[2], b.vect_f32[2]));
    dst.vect_f32[3] = vreinterpretq_f32_u32(g_FunListCmp256Ps[imm8].cmpFun(a.vect_f32[3], b.vect_f32[3]));
    return dst;
}

FORCE_INLINE __mmask16 _mm512_cmp_ps_mask(__m512 a, __m512 b, const int imm8)
{
    assert(imm8 < 32 && imm8 >= 0);
    __m512 dst = _mm512_cmp_ps(a, b, imm8);
    __mmask16 res = 0;
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32);
    uint32_t r0 = vaddvq_u32(vandq_u32(vreinterpretq_u32_f32(dst.vect_f32[0]), vect_mask));
    uint32_t r1 = vaddvq_u32(vandq_u32(vreinterpretq_u32_f32(dst.vect_f32[1]), vect_mask));
    uint32_t r2 = vaddvq_u32(vandq_u32(vreinterpretq_u32_f32(dst.vect_f32[2]), vect_mask));
    uint32_t r3 = vaddvq_u32(vandq_u32(vreinterpretq_u32_f32(dst.vect_f32[3]), vect_mask));
    res = r0 | (r1 << 4) | (r2 << 8) | (r3 << 12);
    return res;
}

FORCE_INLINE __mmask16 _mm512_cmpeq_epi32_mask(__m512i a, __m512i b)
{
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32);
    __m512i tmp;
    tmp.vect_u32[0] = vandq_u32(vceqq_s32(a.vect_s32[0], b.vect_s32[0]), vect_mask);
    tmp.vect_u32[1] = vandq_u32(vceqq_s32(a.vect_s32[1], b.vect_s32[1]), vect_mask);
    tmp.vect_u32[2] = vandq_u32(vceqq_s32(a.vect_s32[2], b.vect_s32[2]), vect_mask);
    tmp.vect_u32[3] = vandq_u32(vceqq_s32(a.vect_s32[3], b.vect_s32[3]), vect_mask);
    uint32_t r0 = vaddvq_u32(tmp.vect_u32[0]);
    uint32_t r1 = vaddvq_u32(tmp.vect_u32[1]);
    uint32_t r2 = vaddvq_u32(tmp.vect_u32[2]);
    uint32_t r3 = vaddvq_u32(tmp.vect_u32[3]);
    __mmask16 result = r0 | (r1 << 4) | (r2 << 8) | (r3 << 12);
    return result;
}

FORCE_INLINE __mmask16 _mm512_cmplt_epi32_mask(__m512i a, __m512i b)
{
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32);
    __m512i tmp;
    tmp.vect_u32[0] = vandq_u32(vcltq_s32(a.vect_s32[0], b.vect_s32[0]), vect_mask);
    tmp.vect_u32[1] = vandq_u32(vcltq_s32(a.vect_s32[1], b.vect_s32[1]), vect_mask);
    tmp.vect_u32[2] = vandq_u32(vcltq_s32(a.vect_s32[2], b.vect_s32[2]), vect_mask);
    tmp.vect_u32[3] = vandq_u32(vcltq_s32(a.vect_s32[3], b.vect_s32[3]), vect_mask);
    uint32_t r0 = vaddvq_u32(tmp.vect_u32[0]);
    uint32_t r1 = vaddvq_u32(tmp.vect_u32[1]);
    uint32_t r2 = vaddvq_u32(tmp.vect_u32[2]);
    uint32_t r3 = vaddvq_u32(tmp.vect_u32[3]);
    __mmask16 result = r0 | (r1 << 4) | (r2 << 8) | (r3 << 12);
    return result;
}

FORCE_INLINE __mmask16 _mm512_cmpgt_epi32_mask (__m512i a, __m512i b)
{
    __mmask16 sign;
    __mmask16 *k = &sign;
    __m512i res;
    res.vect_u32[0] = vcgtq_s32(a.vect_s32[0], b.vect_s32[0]);
    res.vect_u32[1] = vcgtq_s32(a.vect_s32[1], b.vect_s32[1]);
    res.vect_u32[2] = vcgtq_s32(a.vect_s32[2], b.vect_s32[2]);
    res.vect_u32[3] = vcgtq_s32(a.vect_s32[3], b.vect_s32[3]);
    PICK_HB_32x16(res, k);
    return sign;
}

FORCE_INLINE __mmask16 _mm512_cmple_epi32_mask(__m512i a, __m512i b)
{
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32);
    __m512i tmp;
    tmp.vect_u32[0] = vandq_u32(vcleq_s32(a.vect_s32[0], b.vect_s32[0]), vect_mask);
    tmp.vect_u32[1] = vandq_u32(vcleq_s32(a.vect_s32[1], b.vect_s32[1]), vect_mask);
    tmp.vect_u32[2] = vandq_u32(vcleq_s32(a.vect_s32[2], b.vect_s32[2]), vect_mask);
    tmp.vect_u32[3] = vandq_u32(vcleq_s32(a.vect_s32[3], b.vect_s32[3]), vect_mask);
    uint32_t r0 = vaddvq_u32(tmp.vect_u32[0]);
    uint32_t r1 = vaddvq_u32(tmp.vect_u32[1]);
    uint32_t r2 = vaddvq_u32(tmp.vect_u32[2]);
    uint32_t r3 = vaddvq_u32(tmp.vect_u32[3]);
    __mmask16 result = r0 | (r1 << 4) | (r2 << 8) | (r3 << 12);
    return result;
}

FORCE_INLINE __mmask16 _mm512_cmpneq_epi32_mask(__m512i a, __m512i b)
{
    return ~_mm512_cmpeq_epi32_mask(a, b);
}

FORCE_INLINE __mmask16 _mm512_cmpnlt_epi32_mask(__m512i a, __m512i b)
{
    return ~_mm512_cmplt_epi32_mask(a, b);
}

FORCE_INLINE __mmask16 _mm512_cmpnle_epi32_mask(__m512i a, __m512i b)
{
    return ~_mm512_cmple_epi32_mask(a, b);
}

typedef __mmask16 (*TYPE_FUNC_EPI32)(__m512i a, __m512i b);
typedef struct {
    _MM_CMPINT_ENUM cmpintEnum;
    TYPE_FUNC_EPI32 cmpFun;
} FuncListEpi32;
static FuncListEpi32 g_FunListEpi32[] = {{_MM_CMPINT_EQ, _mm512_cmpeq_epi32_mask},
    {_MM_CMPINT_LT, _mm512_cmplt_epi32_mask},
    {_MM_CMPINT_LE, _mm512_cmple_epi32_mask},
    {_MM_CMPINT_FALSE, NULL},
    {_MM_CMPINT_NE, _mm512_cmpneq_epi32_mask},
    {_MM_CMPINT_NLT, _mm512_cmpnlt_epi32_mask},
    {_MM_CMPINT_NLE, _mm512_cmpnle_epi32_mask},
    {_MM_CMPINT_TRUE, NULL}};

FORCE_INLINE __mmask16 _mm512_cmp_epi32_mask(__m512i a, __m512i b, const _MM_CMPINT_ENUM imm8)
{
    if (unlikely(imm8 == _MM_CMPINT_FALSE)) {
        return 0;
    }
    if (unlikely(imm8 == _MM_CMPINT_TRUE)) {
        return 0xffff;
    }
    return g_FunListEpi32[imm8].cmpFun(a, b);
}

FORCE_INLINE __mmask64 _mm512_cmpeq_epi8_mask(__m512i a, __m512i b)
{
    uint8x16_t vect_mask = vld1q_u8(g_mask_epi8);
    __m512i tmp;
    tmp.vect_u8[0] = vandq_u8(vceqq_s8(a.vect_s8[0], b.vect_s8[0]), vect_mask);
    tmp.vect_u8[1] = vandq_u8(vceqq_s8(a.vect_s8[1], b.vect_s8[1]), vect_mask);
    tmp.vect_u8[2] = vandq_u8(vceqq_s8(a.vect_s8[2], b.vect_s8[2]), vect_mask);
    tmp.vect_u8[3] = vandq_u8(vceqq_s8(a.vect_s8[3], b.vect_s8[3]), vect_mask);
    uint64_t r0 = vaddv_u8(vget_low_u8(tmp.vect_u8[0])) | (vaddv_u8(vget_high_u8(tmp.vect_u8[0])) << 8);
    uint64_t r1 = vaddv_u8(vget_low_u8(tmp.vect_u8[1])) | (vaddv_u8(vget_high_u8(tmp.vect_u8[1])) << 8);
    uint64_t r2 = vaddv_u8(vget_low_u8(tmp.vect_u8[2])) | (vaddv_u8(vget_high_u8(tmp.vect_u8[2])) << 8);
    uint64_t r3 = vaddv_u8(vget_low_u8(tmp.vect_u8[3])) | (vaddv_u8(vget_high_u8(tmp.vect_u8[3])) << 8);
    __mmask64 result = r0 | (r1 << 16) | (r2 << 32) | (r3 << 48);
    return result;
}

FORCE_INLINE __mmask64 _mm512_mask_cmpeq_epi8_mask(__mmask64 k1, __m512i a, __m512i b)
{
    return (_mm512_cmpeq_epi8_mask(a, b) & k1);
}

FORCE_INLINE __mmask64 _mm512_cmplt_epi8_mask(__m512i a, __m512i b)
{
    uint8x16_t vect_mask = vld1q_u8(g_mask_epi8);
    __m512i tmp;
    tmp.vect_u8[0] = vandq_u8(vcltq_s8(a.vect_s8[0], b.vect_s8[0]), vect_mask);
    tmp.vect_u8[1] = vandq_u8(vcltq_s8(a.vect_s8[1], b.vect_s8[1]), vect_mask);
    tmp.vect_u8[2] = vandq_u8(vcltq_s8(a.vect_s8[2], b.vect_s8[2]), vect_mask);
    tmp.vect_u8[3] = vandq_u8(vcltq_s8(a.vect_s8[3], b.vect_s8[3]), vect_mask);
    uint64_t r0 = vaddv_u8(vget_low_u8(tmp.vect_u8[0])) | (vaddv_u8(vget_high_u8(tmp.vect_u8[0])) << 8);
    uint64_t r1 = vaddv_u8(vget_low_u8(tmp.vect_u8[1])) | (vaddv_u8(vget_high_u8(tmp.vect_u8[1])) << 8);
    uint64_t r2 = vaddv_u8(vget_low_u8(tmp.vect_u8[2])) | (vaddv_u8(vget_high_u8(tmp.vect_u8[2])) << 8);
    uint64_t r3 = vaddv_u8(vget_low_u8(tmp.vect_u8[3])) | (vaddv_u8(vget_high_u8(tmp.vect_u8[3])) << 8);
    __mmask64 result = r0 | (r1 << 16) | (r2 << 32) | (r3 << 48);
    return result;
}

FORCE_INLINE __mmask64 _mm512_cmple_epi8_mask(__m512i a, __m512i b)
{
    uint8x16_t vect_mask = vld1q_u8(g_mask_epi8);
    __m512i tmp;
    tmp.vect_u8[0] = vandq_u8(vcleq_s8(a.vect_s8[0], b.vect_s8[0]), vect_mask);
    tmp.vect_u8[1] = vandq_u8(vcleq_s8(a.vect_s8[1], b.vect_s8[1]), vect_mask);
    tmp.vect_u8[2] = vandq_u8(vcleq_s8(a.vect_s8[2], b.vect_s8[2]), vect_mask);
    tmp.vect_u8[3] = vandq_u8(vcleq_s8(a.vect_s8[3], b.vect_s8[3]), vect_mask);
    uint64_t r0 = vaddv_u8(vget_low_u8(tmp.vect_u8[0])) | (vaddv_u8(vget_high_u8(tmp.vect_u8[0])) << 8);
    uint64_t r1 = vaddv_u8(vget_low_u8(tmp.vect_u8[1])) | (vaddv_u8(vget_high_u8(tmp.vect_u8[1])) << 8);
    uint64_t r2 = vaddv_u8(vget_low_u8(tmp.vect_u8[2])) | (vaddv_u8(vget_high_u8(tmp.vect_u8[2])) << 8);
    uint64_t r3 = vaddv_u8(vget_low_u8(tmp.vect_u8[3])) | (vaddv_u8(vget_high_u8(tmp.vect_u8[3])) << 8);
    __mmask64 result = r0 | (r1 << 16) | (r2 << 32) | (r3 << 48);
    return result;
}

FORCE_INLINE __mmask64 _mm512_cmpneq_epi8_mask(__m512i a, __m512i b)
{
    return ~_mm512_cmpeq_epi8_mask(a, b);
}

FORCE_INLINE __mmask64 _mm512_cmpnlt_epi8_mask(__m512i a, __m512i b)
{
    return ~_mm512_cmplt_epi8_mask(a, b);
}

FORCE_INLINE __mmask64 _mm512_cmpnle_epi8_mask(__m512i a, __m512i b)
{
    return ~_mm512_cmple_epi8_mask(a, b);
}

typedef __mmask64 (*TYPE_FUNC_EPI8)(__m512i a, __m512i b);
typedef struct {
    _MM_CMPINT_ENUM cmpintEnum;
    TYPE_FUNC_EPI8 cmpFun;
} FuncListEpi8;

static FuncListEpi8 g_FunListEpi8[] = {{_MM_CMPINT_EQ, _mm512_cmpeq_epi8_mask},
    {_MM_CMPINT_LT, _mm512_cmplt_epi8_mask},
    {_MM_CMPINT_LE, _mm512_cmple_epi8_mask},
    {_MM_CMPINT_FALSE, NULL},
    {_MM_CMPINT_NE, _mm512_cmpneq_epi8_mask},
    {_MM_CMPINT_NLT, _mm512_cmpnlt_epi8_mask},
    {_MM_CMPINT_NLE, _mm512_cmpnle_epi8_mask},
    {_MM_CMPINT_TRUE, NULL}};

FORCE_INLINE __mmask64 _mm512_cmp_epi8_mask(__m512i a, __m512i b, const int imm8)
{
    if (unlikely(imm8 == _MM_CMPINT_FALSE)) {
        return 0;
    }
    if (unlikely(imm8 == _MM_CMPINT_TRUE)) {
        return 0xffffffffffffffff;
    }
    return g_FunListEpi8[imm8].cmpFun(a, b);
}

FORCE_INLINE __m512i _mm512_and_si512(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s32[0] = vandq_s32(a.vect_s32[0], b.vect_s32[0]);
    res_m512i.vect_s32[1] = vandq_s32(a.vect_s32[1], b.vect_s32[1]);
    res_m512i.vect_s32[2] = vandq_s32(a.vect_s32[2], b.vect_s32[2]);
    res_m512i.vect_s32[3] = vandq_s32(a.vect_s32[3], b.vect_s32[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_or_si512(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s32[0] = vorrq_s32(a.vect_s32[0], b.vect_s32[0]);
    res_m512i.vect_s32[1] = vorrq_s32(a.vect_s32[1], b.vect_s32[1]);
    res_m512i.vect_s32[2] = vorrq_s32(a.vect_s32[2], b.vect_s32[2]);
    res_m512i.vect_s32[3] = vorrq_s32(a.vect_s32[3], b.vect_s32[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_andnot_si512(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s32[0] = vbicq_s32(b.vect_s32[0], a.vect_s32[0]);
    res_m512i.vect_s32[1] = vbicq_s32(b.vect_s32[1], a.vect_s32[1]);
    res_m512i.vect_s32[2] = vbicq_s32(b.vect_s32[2], a.vect_s32[2]);
    res_m512i.vect_s32[3] = vbicq_s32(b.vect_s32[3], a.vect_s32[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_xor_si512(__m512i a, __m512i b)
{
    __m512i res_m512i;
    res_m512i.vect_s32[0] = veorq_s32(a.vect_s32[0], b.vect_s32[0]);
    res_m512i.vect_s32[1] = veorq_s32(a.vect_s32[1], b.vect_s32[1]);
    res_m512i.vect_s32[2] = veorq_s32(a.vect_s32[2], b.vect_s32[2]);
    res_m512i.vect_s32[3] = veorq_s32(a.vect_s32[3], b.vect_s32[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_and_epi32 (__m512i a, __m512i b)
{
    a.vect_s32[0] = vandq_s32(a.vect_s32[0], b.vect_s32[0]);
    a.vect_s32[1] = vandq_s32(a.vect_s32[1], b.vect_s32[1]);
    a.vect_s32[2] = vandq_s32(a.vect_s32[2], b.vect_s32[2]);
    a.vect_s32[3] = vandq_s32(a.vect_s32[3], b.vect_s32[3]);
    return a;
}

FORCE_INLINE __m512i _mm512_and_epi64 (__m512i a, __m512i b)
{
    a.vect_s64[0] = vandq_s64(a.vect_s64[0], b.vect_s64[0]);
    a.vect_s64[1] = vandq_s64(a.vect_s64[1], b.vect_s64[1]);
    a.vect_s64[2] = vandq_s64(a.vect_s64[2], b.vect_s64[2]);
    a.vect_s64[3] = vandq_s64(a.vect_s64[3], b.vect_s64[3]);
    return a;
}

FORCE_INLINE __m512i _mm512_or_epi32 (__m512i a, __m512i b)
{
    a.vect_s32[0] = vorrq_s32(a.vect_s32[0], b.vect_s32[0]);
    a.vect_s32[1] = vorrq_s32(a.vect_s32[1], b.vect_s32[1]);
    a.vect_s32[2] = vorrq_s32(a.vect_s32[2], b.vect_s32[2]);
    a.vect_s32[3] = vorrq_s32(a.vect_s32[3], b.vect_s32[3]);
    return a;
}

FORCE_INLINE __m512i _mm512_or_epi64 (__m512i a, __m512i b)
{
    a.vect_s64[0] = vorrq_s64(a.vect_s64[0], b.vect_s64[0]);
    a.vect_s64[1] = vorrq_s64(a.vect_s64[1], b.vect_s64[1]);
    a.vect_s64[2] = vorrq_s64(a.vect_s64[2], b.vect_s64[2]);
    a.vect_s64[3] = vorrq_s64(a.vect_s64[3], b.vect_s64[3]);
    return a;
}

FORCE_INLINE __m512 _mm512_xor_ps (__m512 a, __m512 b)
{
    __asm__ __volatile(
        "eor %0.16b, %0.16b, %2.16b     \n\t"
        "eor %1.16b, %1.16b, %3.16b     \n\t"
        :"+w"(a.vect_f32[0]), "+w"(a.vect_f32[1])
        :"w"(b.vect_f32[0]), "w"(b.vect_f32[1])
    );
    __asm__ __volatile(
        "eor %0.16b, %0.16b, %2.16b     \n\t"
        "eor %1.16b, %1.16b, %3.16b     \n\t"
        :"+w"(a.vect_f32[2]), "+w"(a.vect_f32[3])
        :"w"(b.vect_f32[2]), "w"(b.vect_f32[3])
    );
    return a;
}

FORCE_INLINE __m512d _mm512_xor_pd (__m512d a, __m512d b)
{
    __asm__ __volatile(
        "eor %0.16b, %0.16b, %2.16b     \n\t"
        "eor %1.16b, %1.16b, %3.16b     \n\t"
        :"+w"(a.vect_f64[0]), "+w"(a.vect_f64[1])
        :"w"(b.vect_f64[0]), "w"(b.vect_f64[1])
    );
    __asm__ __volatile(
        "eor %0.16b, %0.16b, %2.16b     \n\t"
        "eor %1.16b, %1.16b, %3.16b     \n\t"
        :"+w"(a.vect_f64[2]), "+w"(a.vect_f64[3])
        :"w"(b.vect_f64[2]), "w"(b.vect_f64[3])
    );
    return a;
}

FORCE_INLINE __m512i _mm512_set_epi32(int e15, int e14, int e13, int e12, int e11, int e10, int e9, int e8, int e7,
    int e6, int e5, int e4, int e3, int e2, int e1, int e0)
{
    __m512i res_m512i;
    SET32x4(res_m512i.vect_s32[0], e0, e1, e2, e3);
    SET32x4(res_m512i.vect_s32[1], e4, e5, e6, e7);
    SET32x4(res_m512i.vect_s32[2], e8, e9, e10, e11);
    SET32x4(res_m512i.vect_s32[3], e12, e13, e14, e15);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_set_epi64(
    __int64 e7, __int64 e6, __int64 e5, __int64 e4, __int64 e3, __int64 e2, __int64 e1, __int64 e0)
{
    __m512i res_m512i;
    SET64x2(res_m512i.vect_s64[0], e0, e1);
    SET64x2(res_m512i.vect_s64[1], e2, e3);
    SET64x2(res_m512i.vect_s64[2], e4, e5);
    SET64x2(res_m512i.vect_s64[3], e6, e7);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_set1_epi32(int a)
{
    __m512i res_m512i;
    res_m512i.vect_s32[0] = vdupq_n_s32(a);
    res_m512i.vect_s32[1] = vdupq_n_s32(a);
    res_m512i.vect_s32[2] = vdupq_n_s32(a);
    res_m512i.vect_s32[3] = vdupq_n_s32(a);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_set1_epi64(__int64 a)
{
    __m512i res_m512i;
    res_m512i.vect_s64[0] = vdupq_n_s64(a);
    res_m512i.vect_s64[1] = vdupq_n_s64(a);
    res_m512i.vect_s64[2] = vdupq_n_s64(a);
    res_m512i.vect_s64[3] = vdupq_n_s64(a);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_set1_epi8(char a)
{
    __m512i res_m512i;
    res_m512i.vect_s8[0] = vdupq_n_s8(a);
    res_m512i.vect_s8[1] = vdupq_n_s8(a);
    res_m512i.vect_s8[2] = vdupq_n_s8(a);
    res_m512i.vect_s8[3] = vdupq_n_s8(a);
    return res_m512i;
}

FORCE_INLINE __m512 _mm512_set_ps(float e15, float e14, float e13, float e12, float e11, float e10, float e9, float e8,
    float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0)
{
    __m512 res_m512;
    SET32x4(res_m512.vect_f32[0], e0, e1, e2, e3);
    SET32x4(res_m512.vect_f32[1], e4, e5, e6, e7);
    SET32x4(res_m512.vect_f32[2], e8, e9, e10, e11);
    SET32x4(res_m512.vect_f32[3], e12, e13, e14, e15);
    return res_m512;
}

FORCE_INLINE __m512d _mm512_set_pd(
    double e7, double e6, double e5, double e4, double e3, double e2, double e1, double e0)
{
    __m512d res_m512d;
    SET64x2(res_m512d.vect_f64[0], e0, e1);
    SET64x2(res_m512d.vect_f64[1], e2, e3);
    SET64x2(res_m512d.vect_f64[2], e4, e5);
    SET64x2(res_m512d.vect_f64[3], e6, e7);
    return res_m512d;
}

FORCE_INLINE __m512 _mm512_set1_ps(float a)
{
    __m512 res_m512;
    res_m512.vect_f32[0] = vdupq_n_f32(a);
    res_m512.vect_f32[1] = vdupq_n_f32(a);
    res_m512.vect_f32[2] = vdupq_n_f32(a);
    res_m512.vect_f32[3] = vdupq_n_f32(a);
    return res_m512;
}

FORCE_INLINE __m512d _mm512_set1_pd(double a)
{
    __m512d res_m512d;
    res_m512d.vect_f64[0] = vdupq_n_f64(a);
    res_m512d.vect_f64[1] = vdupq_n_f64(a);
    res_m512d.vect_f64[2] = vdupq_n_f64(a);
    res_m512d.vect_f64[3] = vdupq_n_f64(a);
    return res_m512d;
}

FORCE_INLINE __m512 _mm512_setzero_ps()
{
    __m512 res_m512;
    res_m512.vect_f32[0] = vdupq_n_f32(0.0);
    res_m512.vect_f32[1] = vdupq_n_f32(0.0);
    res_m512.vect_f32[2] = vdupq_n_f32(0.0);
    res_m512.vect_f32[3] = vdupq_n_f32(0.0);
    return res_m512;
}

FORCE_INLINE __m512d _mm512_setzero_pd()
{
    __m512d res_m512d;
    res_m512d.vect_f64[0] = vdupq_n_f64(0.0);
    res_m512d.vect_f64[1] = vdupq_n_f64(0.0);
    res_m512d.vect_f64[2] = vdupq_n_f64(0.0);
    res_m512d.vect_f64[3] = vdupq_n_f64(0.0);
    return res_m512d;
}

FORCE_INLINE __m512i _mm512_setzero_si512 ()
{
    __m512i res_m512i;
    res_m512i.vect_s32[0] = vdupq_n_s32(0);
    res_m512i.vect_s32[1] = res_m512i.vect_s32[0];
    res_m512i.vect_s32[2] = res_m512i.vect_s32[0];
    res_m512i.vect_s32[3] = res_m512i.vect_s32[0];
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_movm_epi8(__mmask64 k)
{
    uint8x8_t mk = vcreate_u8(k);
    uint8x16_t mask_and = vld1q_u8(g_mask_epi8);

    __m512i res_m512i;
    res_m512i.vect_u8[0] = vcombine_u8(vdup_lane_u8(mk, 0), vdup_lane_u8(mk, 1));
    res_m512i.vect_u8[1] = vcombine_u8(vdup_lane_u8(mk, 2), vdup_lane_u8(mk, 3));
    res_m512i.vect_u8[2] = vcombine_u8(vdup_lane_u8(mk, 4), vdup_lane_u8(mk, 5));
    res_m512i.vect_u8[3] = vcombine_u8(vdup_lane_u8(mk, 6), vdup_lane_u8(mk, 7));
    res_m512i.vect_u8[0] = vtstq_u8(mask_and, res_m512i.vect_u8[0]);
    res_m512i.vect_u8[1] = vtstq_u8(mask_and, res_m512i.vect_u8[1]);
    res_m512i.vect_u8[2] = vtstq_u8(mask_and, res_m512i.vect_u8[2]);
    res_m512i.vect_u8[3] = vtstq_u8(mask_and, res_m512i.vect_u8[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_movm_epi32 (__mmask16 k)
{
    __m512i res;
    unsigned int mk = k;
    uint32x4_t mask_and = vld1q_u32(g_mask_epi32);
    res.vect_u32[0] = vtstq_u32(vdupq_n_u32(mk), mask_and);
    res.vect_u32[1] = vtstq_u32(vdupq_n_u32(mk >> 4), mask_and);
    res.vect_u32[2] = vtstq_u32(vdupq_n_u32(mk >> 8), mask_and);
    res.vect_u32[3] = vtstq_u32(vdupq_n_u32(mk >> 12), mask_and);
    return res;
}

FORCE_INLINE __m128i _mm512_extracti32x4_epi32(__m512i a, const int imm8)
{
    assert(imm8 >= 0 && imm8 <= 3);
    __m128i res_m128i;
    res_m128i.vect_s32 = a.vect_s32[imm8];
    return res_m128i;
}

FORCE_INLINE __m256 _mm512_extractf32x8_ps (__m512 a, int imm8)
{
    assert(imm8 >= 0 && imm8 <= 1);
    __m256 res_m256;
    int id = imm8 << 1;
    res_m256.vect_f32[0] = a.vect_f32[id];
    res_m256.vect_f32[1] = a.vect_f32[id | 1];
    return res_m256;
}

FORCE_INLINE __m256d _mm512_extractf64x4_pd (__m512d a, int imm8)
{
    assert(imm8 >= 0 && imm8 <= 1);
    __m256d res_m256d;
    int id = imm8 << 1;
    res_m256d.vect_f64[0] = a.vect_f64[id];
    res_m256d.vect_f64[1] = a.vect_f64[id | 1];
    return res_m256d;
}

FORCE_INLINE void _mm512_store_si512(void* mem_addr, __m512i a)
{
    vst1q_s64((int64_t*)mem_addr, a.vect_s64[0]);
    vst1q_s64((int64_t*)mem_addr + 2, a.vect_s64[1]);
    vst1q_s64((int64_t*)mem_addr + 4, a.vect_s64[2]);
    vst1q_s64((int64_t*)mem_addr + 6, a.vect_s64[3]);
}

FORCE_INLINE void _mm512_storeu_si512 (void* mem_addr, __m512i a)
{
    vst1q_s64((int64_t*)mem_addr, a.vect_s64[0]);
    vst1q_s64((int64_t*)mem_addr + 2, a.vect_s64[1]);
    vst1q_s64((int64_t*)mem_addr + 4, a.vect_s64[2]);
    vst1q_s64((int64_t*)mem_addr + 6, a.vect_s64[3]);
}

FORCE_INLINE void _mm512_stream_si512 (void* mem_addr, __m512i a)
{
    vst1q_s64((int64_t*)mem_addr, a.vect_s64[0]);
    vst1q_s64((int64_t*)mem_addr + 2, a.vect_s64[1]);
    vst1q_s64((int64_t*)mem_addr + 4, a.vect_s64[2]);
    vst1q_s64((int64_t*)mem_addr + 6, a.vect_s64[3]);
}

FORCE_INLINE __m512i _mm512_load_si512(void const* mem_addr)
{
    __m512i ret;
    ret.vect_s32[0] = vld1q_s32((int32_t const*)mem_addr);
    ret.vect_s32[1] = vld1q_s32(((int32_t const*)mem_addr) + 4);
    ret.vect_s32[2] = vld1q_s32(((int32_t const*)mem_addr) + 8);
    ret.vect_s32[3] = vld1q_s32(((int32_t const*)mem_addr) + 12);
    return ret;
}

FORCE_INLINE __m512i _mm512_loadu_si512(void const* mem_addr)
{
    __m512i ret;
    ret.vect_s32[0] = vld1q_s32((int32_t const*)mem_addr);
    ret.vect_s32[1] = vld1q_s32(((int32_t const*)mem_addr) + 4);
    ret.vect_s32[2] = vld1q_s32(((int32_t const*)mem_addr) + 8);
    ret.vect_s32[3] = vld1q_s32(((int32_t const*)mem_addr) + 12);
    return ret;
}

FORCE_INLINE __m512i _mm512_mask_loadu_epi8(__m512i src, __mmask64 k, void const* mem_addr)
{
    __m512i ret;
    int8_t const* data_addr = (int8_t const*)mem_addr;
    uint8x16_t mask = vld1q_u8(g_mask_epi8);
    uint8x16_t k_vec[4];
    k_vec[0] = vcombine_u8(vdup_n_u8(k & 0xff), vdup_n_u8((k >> 8) & 0xff));
    k_vec[1] = vcombine_u8(vdup_n_u8((k >> 16) & 0xff), vdup_n_u8((k >> 24) & 0xff));
    k_vec[2] = vcombine_u8(vdup_n_u8((k >> 32) & 0xff), vdup_n_u8((k >> 40) & 0xff));
    k_vec[3] = vcombine_u8(vdup_n_u8((k >> 48) & 0xff), vdup_n_u8((k >> 56) & 0xff));
    ret.vect_s8[0] = vbslq_s8(vtstq_u8(k_vec[0], mask), vld1q_s8(data_addr), src.vect_s8[0]);
    ret.vect_s8[1] = vbslq_s8(vtstq_u8(k_vec[1], mask), vld1q_s8(data_addr + 16), src.vect_s8[1]);
    ret.vect_s8[2] = vbslq_s8(vtstq_u8(k_vec[2], mask), vld1q_s8(data_addr + 32), src.vect_s8[2]);
    ret.vect_s8[3] = vbslq_s8(vtstq_u8(k_vec[3], mask), vld1q_s8(data_addr + 48), src.vect_s8[3]);
    return ret;
}

FORCE_INLINE __m512i _mm512_maskz_loadu_epi8(__mmask64 k, void const* mem_addr)
{
    __m512i ret;
    uint8_t const* data_addr = (uint8_t const*)mem_addr;
    uint8x16_t mask = vld1q_u8(g_mask_epi8);
    uint8x16_t k_vec[4];
    k_vec[0] = vcombine_u8(vdup_n_u8(k & 0xff), vdup_n_u8((k >> 8) & 0xff));
    k_vec[1] = vcombine_u8(vdup_n_u8((k >> 16) & 0xff), vdup_n_u8((k >> 24) & 0xff));
    k_vec[2] = vcombine_u8(vdup_n_u8((k >> 32) & 0xff), vdup_n_u8((k >> 40) & 0xff));
    k_vec[3] = vcombine_u8(vdup_n_u8((k >> 48) & 0xff), vdup_n_u8((k >> 56) & 0xff));
    ret.vect_u8[0] = vandq_u8(vtstq_u8(k_vec[0], mask), vld1q_u8(data_addr));
    ret.vect_u8[1] = vandq_u8(vtstq_u8(k_vec[1], mask), vld1q_u8(data_addr + 16));
    ret.vect_u8[2] = vandq_u8(vtstq_u8(k_vec[2], mask), vld1q_u8(data_addr + 32));
    ret.vect_u8[3] = vandq_u8(vtstq_u8(k_vec[3], mask), vld1q_u8(data_addr + 48));
    return ret;
}

FORCE_INLINE __m512i _mm512_abs_epi8(__m512i a)
{
    __m512i ret;
    ret.vect_s8[0] = vabsq_s8(a.vect_s8[0]);
    ret.vect_s8[1] = vabsq_s8(a.vect_s8[1]);
    ret.vect_s8[2] = vabsq_s8(a.vect_s8[2]);
    ret.vect_s8[3] = vabsq_s8(a.vect_s8[3]);
    return ret;
}

FORCE_INLINE __m512i _mm512_broadcast_i32x4(__m128i a)
{
    __m512i ret;
    ret.vect_s32[0] = a.vect_s32;
    ret.vect_s32[1] = a.vect_s32;
    ret.vect_s32[2] = a.vect_s32;
    ret.vect_s32[3] = a.vect_s32;
    return ret;
}

FORCE_INLINE __m512i _mm512_broadcast_i64x4(__m256i a)
{
    __m512i ret;
    ret.vect_s64[0] = a.vect_s64[0];
    ret.vect_s64[1] = a.vect_s64[1];
    ret.vect_s64[2] = a.vect_s64[0];
    ret.vect_s64[3] = a.vect_s64[1];
    return ret;
}

FORCE_INLINE __m512i _mm512_mask_broadcast_i64x4(__m512i src, __mmask8 k, __m256i a)
{
    __m512i ret;
    uint64x2_t vect_mask = vld1q_u64(g_mask_epi64);
    uint64x2_t tmp[4];
    tmp[0] = vtstq_u64(vdupq_n_u64(k & 0x03), vect_mask);
    tmp[1] = vtstq_u64(vdupq_n_u64((k & 0x0c) >> 2), vect_mask);
    tmp[2] = vtstq_u64(vdupq_n_u64((k & 0x30) >> 4), vect_mask);
    tmp[3] = vtstq_u64(vdupq_n_u64((k & 0xc0) >> 6), vect_mask);
    ret.vect_s64[0] = vbslq_s64(tmp[0], a.vect_s64[0], src.vect_s64[0]);
    ret.vect_s64[1] = vbslq_s64(tmp[1], a.vect_s64[1], src.vect_s64[1]);
    ret.vect_s64[2] = vbslq_s64(tmp[2], a.vect_s64[0], src.vect_s64[2]);
    ret.vect_s64[3] = vbslq_s64(tmp[3], a.vect_s64[1], src.vect_s64[3]);
    return ret;
}

FORCE_INLINE __m512i _mm512_shuffle_epi8(__m512i a, __m512i b)
{
    __m512i res_m512i;
    uint8x16_t mask_and = vdupq_n_u8(0x8f);
    res_m512i.vect_u8[0] = vqtbl1q_u8(a.vect_u8[0], vandq_u8(b.vect_u8[0], mask_and));
    res_m512i.vect_u8[1] = vqtbl1q_u8(a.vect_u8[1], vandq_u8(b.vect_u8[1], mask_and));
    res_m512i.vect_u8[2] = vqtbl1q_u8(a.vect_u8[2], vandq_u8(b.vect_u8[2], mask_and));
    res_m512i.vect_u8[3] = vqtbl1q_u8(a.vect_u8[3], vandq_u8(b.vect_u8[3], mask_and));
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_maskz_shuffle_epi8(__mmask64 k, __m512i a, __m512i b)
{
    uint8x8_t mk = vcreate_u8(k);
    uint8x16_t mask_and = vld1q_u8(g_mask_epi8);

    __m512i tmp, res_m512i;
    tmp.vect_u8[0] = vtstq_u8(mask_and, vcombine_u8(vdup_lane_u8(mk, 0), vdup_lane_u8(mk, 1)));
    tmp.vect_u8[1] = vtstq_u8(mask_and, vcombine_u8(vdup_lane_u8(mk, 2), vdup_lane_u8(mk, 3)));
    tmp.vect_u8[2] = vtstq_u8(mask_and, vcombine_u8(vdup_lane_u8(mk, 4), vdup_lane_u8(mk, 5)));
    tmp.vect_u8[3] = vtstq_u8(mask_and, vcombine_u8(vdup_lane_u8(mk, 6), vdup_lane_u8(mk, 7)));
    mask_and = vdupq_n_u8(0x8f);
    res_m512i.vect_u8[0] = vqtbl1q_u8(a.vect_u8[0], vandq_u8(b.vect_u8[0], mask_and));
    res_m512i.vect_u8[1] = vqtbl1q_u8(a.vect_u8[1], vandq_u8(b.vect_u8[1], mask_and));
    res_m512i.vect_u8[2] = vqtbl1q_u8(a.vect_u8[2], vandq_u8(b.vect_u8[2], mask_and));
    res_m512i.vect_u8[3] = vqtbl1q_u8(a.vect_u8[3], vandq_u8(b.vect_u8[3], mask_and));
    res_m512i.vect_u8[0] = vminq_u8(res_m512i.vect_u8[0], tmp.vect_u8[0]);
    res_m512i.vect_u8[1] = vminq_u8(res_m512i.vect_u8[1], tmp.vect_u8[1]);
    res_m512i.vect_u8[2] = vminq_u8(res_m512i.vect_u8[2], tmp.vect_u8[2]);
    res_m512i.vect_u8[3] = vminq_u8(res_m512i.vect_u8[3], tmp.vect_u8[3]);
    return res_m512i;
}

FORCE_INLINE __m512i _mm512_multishift_epi64_epi8(__m512i a, __m512i b)
{
    __m512i res;
    res.vect_i256[0] = _mm256_multishift_epi64_epi8(a.vect_i256[0], b.vect_i256[0]);
    res.vect_i256[1] = _mm256_multishift_epi64_epi8(a.vect_i256[1], b.vect_i256[1]);
    return res;
}

FORCE_INLINE __m512i _mm512_mask_blend_epi32 (__mmask16 k, __m512i a, __m512i b)
{
    __m512i res;
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32);
    uint32x4_t vect_imm = vdupq_n_u32(k);
    uint32x4_t flag[4];
    flag[0] = vtstq_u32(vect_imm, vect_mask);
    flag[1] = vtstq_u32(vshrq_n_u32(vect_imm, 4), vect_mask);
    flag[2] = vtstq_u32(vshrq_n_u32(vect_imm, 8), vect_mask);
    flag[3] = vtstq_u32(vshrq_n_u32(vect_imm, 12), vect_mask);
    res.vect_s32[0] = vbslq_s32(flag[0], b.vect_s32[0], a.vect_s32[0]);
    res.vect_s32[1] = vbslq_s32(flag[1], b.vect_s32[1], a.vect_s32[1]);
    res.vect_s32[2] = vbslq_s32(flag[2], b.vect_s32[2], a.vect_s32[2]);
    res.vect_s32[3] = vbslq_s32(flag[3], b.vect_s32[3], a.vect_s32[3]);
    return res;
}

FORCE_INLINE __m512 _mm512_mask_blend_ps(__mmask16 k, __m512 a, __m512 b)
{
    __m512 result_m512;
    uint32x4_t vect_mask = vld1q_u32(g_mask_epi32);
    uint32x4_t vect_imm = vdupq_n_u32(k);
    uint32x4_t flag[4];
    flag[0] = vtstq_u32(vect_imm, vect_mask);
    flag[1] = vtstq_u32(vshrq_n_u32(vect_imm, 4), vect_mask);
    flag[2] = vtstq_u32(vshrq_n_u32(vect_imm, 8), vect_mask);
    flag[3] = vtstq_u32(vshrq_n_u32(vect_imm, 12), vect_mask);
    result_m512.vect_f32[0] = vbslq_f32(flag[0], b.vect_f32[0], a.vect_f32[0]);
    result_m512.vect_f32[1] = vbslq_f32(flag[1], b.vect_f32[1], a.vect_f32[1]);
    result_m512.vect_f32[2] = vbslq_f32(flag[2], b.vect_f32[2], a.vect_f32[2]);
    result_m512.vect_f32[3] = vbslq_f32(flag[3], b.vect_f32[3], a.vect_f32[3]);
    return result_m512;
}

FORCE_INLINE __m512d _mm512_mask_blend_pd(__mmask8 k, __m512d a, __m512d b)
{
    __m512d result_m512d;
    uint64x2_t vect_mask = vld1q_u64(g_mask_epi64);
    uint64x2_t vect_imm = vdupq_n_u64(k);
    uint64x2_t flag[4];
    flag[0] = vtstq_u64(vect_imm, vect_mask);
    flag[1] = vtstq_u64(vshrq_n_u64(vect_imm, 2), vect_mask);
    flag[2] = vtstq_u64(vshrq_n_u64(vect_imm, 4), vect_mask);
    flag[3] = vtstq_u64(vshrq_n_u64(vect_imm, 6), vect_mask);
    result_m512d.vect_f64[0] = vbslq_f64(flag[0], b.vect_f64[0], a.vect_f64[0]);
    result_m512d.vect_f64[1] = vbslq_f64(flag[1], b.vect_f64[1], a.vect_f64[1]);
    result_m512d.vect_f64[2] = vbslq_f64(flag[2], b.vect_f64[2], a.vect_f64[2]);  
    result_m512d.vect_f64[3] = vbslq_f64(flag[3], b.vect_f64[3], a.vect_f64[3]);
    return result_m512d;
}

FORCE_INLINE __m512d _mm512_castpd128_pd512 (__m128d a)
{
    __m512d ret;
    ret.vect_f64[0] = a;
    return ret;
}

FORCE_INLINE __m128d _mm512_castpd512_pd128 (__m512d a)
{
    return a.vect_f64[0];
}

FORCE_INLINE __m512 _mm512_castps128_ps512 (__m128 a)
{
    __m512 ret;
    ret.vect_f32[0] = a;
    return ret;
}

FORCE_INLINE __m128 _mm512_castps512_ps128 (__m512 a)
{
    return a.vect_f32[0];
}

FORCE_INLINE __m512 _mm512_cvtepi32_ps (__m512i a)
{
    __m512 ret;
    ret.vect_f32[0] = vcvtq_f32_s32(a.vect_s32[0]);
    ret.vect_f32[1] = vcvtq_f32_s32(a.vect_s32[1]);
    ret.vect_f32[2] = vcvtq_f32_s32(a.vect_s32[2]);
    ret.vect_f32[3] = vcvtq_f32_s32(a.vect_s32[3]);
    return ret;
}

FORCE_INLINE __m512d _mm512_cvtepi32_pd (__m256i a)
{
    __m512d res;
    __asm__ __volatile__ (
        "scvtf v0.4s, %[a0].4s           \n\t"
        "scvtf v1.4s, %[a1].4s           \n\t"
        "fcvtl %[r0].2d, v0.2s           \n\t"
        "fcvtl %[r2].2d, v1.2s           \n\t"
        "mov v0.d[0], v0.d[1]            \n\t"
        "mov v1.d[0], v1.d[1]            \n\t"
        "fcvtl %[r1].2d, v0.2s           \n\t"
        "fcvtl %[r3].2d, v1.2s           \n\t"
        :[r0]"=w"(res.vect_f64[0]), [r1]"=w"(res.vect_f64[1]), [r2]"=w"(res.vect_f64[2]), [r3]"=w"(res.vect_f64[3])
        :[a0]"w"(a.vect_s32[0]), [a1]"w"(a.vect_s32[1])
        :"v0", "v1"
    );
    return res;
}

FORCE_INLINE __m512 _mm512_insertf32x8 (__m512 a, __m256 b, int imm8)
{
    assert(imm8 == 0 || imm8 == 1);
    __m512 res;
    uint32x4_t vmask = vceqq_s32(vdupq_n_s32(imm8), vdupq_n_s32(0));
    res.vect_f32[0] = vbslq_f32(vmask, b.vect_f32[0], a.vect_f32[0]);
    res.vect_f32[1] = vbslq_f32(vmask, b.vect_f32[1], a.vect_f32[1]);
    res.vect_f32[2] = vbslq_f32(vmask, a.vect_f32[2], b.vect_f32[0]);
    res.vect_f32[3] = vbslq_f32(vmask, a.vect_f32[3], b.vect_f32[1]);
    return res;
}

FORCE_INLINE __m512d _mm512_insertf64x4 (__m512d a, __m256d b, int imm8)
{
    assert(imm8 == 0 || imm8 == 1);
    __m512d res;
    uint64x2_t vmask = vceqq_s64(vdupq_n_s64(imm8), vdupq_n_s64(0));
    res.vect_f64[0] = vbslq_f64(vmask, b.vect_f64[0], a.vect_f64[0]);
    res.vect_f64[1] = vbslq_f64(vmask, b.vect_f64[1], a.vect_f64[1]);
    res.vect_f64[2] = vbslq_f64(vmask, a.vect_f64[2], b.vect_f64[0]);
    res.vect_f64[3] = vbslq_f64(vmask, a.vect_f64[3], b.vect_f64[1]);
    return res;
}

FORCE_INLINE __m512i _mm512_inserti32x8 (__m512i a, __m256i b, int imm8)
{
    assert(imm8 == 0 || imm8 == 1);
    __m512i res;
    uint32x4_t vmask = vceqq_s32(vdupq_n_s32(imm8), vdupq_n_s32(0));
    res.vect_s32[0] = vbslq_s32(vmask, b.vect_s32[0], a.vect_s32[0]);
    res.vect_s32[1] = vbslq_s32(vmask, b.vect_s32[1], a.vect_s32[1]);
    res.vect_s32[2] = vbslq_s32(vmask, a.vect_s32[2], b.vect_s32[0]);
    res.vect_s32[3] = vbslq_s32(vmask, a.vect_s32[3], b.vect_s32[1]);
    return res;
}

FORCE_INLINE __m512i _mm512_inserti64x4 (__m512i a, __m256i b, int imm8)
{
    assert(imm8 == 0 || imm8 == 1);
    __m512i res;
    uint64x2_t vmask = vceqq_s64(vdupq_n_s64(imm8), vdupq_n_s64(0));
    res.vect_s64[0] = vbslq_s64(vmask, b.vect_s64[0], a.vect_s64[0]);
    res.vect_s64[1] = vbslq_s64(vmask, b.vect_s64[1], a.vect_s64[1]);
    res.vect_s64[2] = vbslq_s64(vmask, a.vect_s64[2], b.vect_s64[0]);
    res.vect_s64[3] = vbslq_s64(vmask, a.vect_s64[3], b.vect_s64[1]);
    return res;
}

FORCE_INLINE __m512i _mm512_load_epi32 (void const* mem_addr)
{
    __m512i res;
    res.vect_s32[0] = vld1q_s32((const int32_t *)mem_addr);
    res.vect_s32[1] = vld1q_s32((const int32_t *)mem_addr + 4);
    res.vect_s32[2] = vld1q_s32((const int32_t *)mem_addr + 8);
    res.vect_s32[3] = vld1q_s32((const int32_t *)mem_addr + 12);
    return res;
}

FORCE_INLINE __m512i _mm512_load_epi64 (void const* mem_addr)
{
    __m512i res;
    res.vect_s64[0] = vld1q_s64((const int64_t *)mem_addr);
    res.vect_s64[1] = vld1q_s64((const int64_t *)mem_addr + 2);
    res.vect_s64[2] = vld1q_s64((const int64_t *)mem_addr + 4);
    res.vect_s64[3] = vld1q_s64((const int64_t *)mem_addr + 6);
    return res;
}

FORCE_INLINE __m512d _mm512_load_pd (void const* mem_addr)
{
    __m512d res;
    res.vect_f64[0] = vld1q_f64((const double *)mem_addr);
    res.vect_f64[1] = vld1q_f64((const double *)mem_addr + 2);
    res.vect_f64[2] = vld1q_f64((const double *)mem_addr + 4);
    res.vect_f64[3] = vld1q_f64((const double *)mem_addr + 6);
    return res;
}

FORCE_INLINE __m512 _mm512_load_ps (void const* mem_addr)
{
    __m512 res;
    res.vect_f32[0] = vld1q_f32((const float *)mem_addr);
    res.vect_f32[1] = vld1q_f32((const float *)mem_addr + 4);
    res.vect_f32[2] = vld1q_f32((const float *)mem_addr + 8);
    res.vect_f32[3] = vld1q_f32((const float *)mem_addr + 12);
    return res;
}

FORCE_INLINE void _mm512_store_epi32 (void* mem_addr, __m512i a)
{
    vst1q_s32((int32_t *)mem_addr, a.vect_s32[0]);
    vst1q_s32((int32_t *)mem_addr + 4, a.vect_s32[1]);
    vst1q_s32((int32_t *)mem_addr + 8, a.vect_s32[2]);
    vst1q_s32((int32_t *)mem_addr + 12, a.vect_s32[3]);
}

FORCE_INLINE void _mm512_store_epi64 (void* mem_addr, __m512i a)
{
    vst1q_s64((int64_t *)mem_addr, a.vect_s64[0]);
    vst1q_s64((int64_t *)mem_addr + 2, a.vect_s64[1]);
    vst1q_s64((int64_t *)mem_addr + 4, a.vect_s64[2]);
    vst1q_s64((int64_t *)mem_addr + 6, a.vect_s64[3]);
}

FORCE_INLINE void _mm512_store_pd (void* mem_addr, __m512d a)
{
    vst1q_f64((float64_t *)mem_addr, a.vect_f64[0]);
    vst1q_f64((float64_t *)mem_addr + 2, a.vect_f64[1]);
    vst1q_f64((float64_t *)mem_addr + 4, a.vect_f64[2]);
    vst1q_f64((float64_t *)mem_addr + 6, a.vect_f64[3]);
}

FORCE_INLINE void _mm512_store_ps (void* mem_addr, __m512 a)
{
    vst1q_f32((float32_t *)mem_addr, a.vect_f32[0]);
    vst1q_f32((float32_t *)mem_addr + 4, a.vect_f32[1]);
    vst1q_f32((float32_t *)mem_addr + 8, a.vect_f32[2]);
    vst1q_f32((float32_t *)mem_addr + 12, a.vect_f32[3]);
}

FORCE_INLINE __m512i _mm512_max_epi32 (__m512i a, __m512i b)
{
    __m512i res;
    res.vect_s32[0] = vmaxq_s32(a.vect_s32[0], b.vect_s32[0]);
    res.vect_s32[1] = vmaxq_s32(a.vect_s32[1], b.vect_s32[1]);
    res.vect_s32[2] = vmaxq_s32(a.vect_s32[2], b.vect_s32[2]);
    res.vect_s32[3] = vmaxq_s32(a.vect_s32[3], b.vect_s32[3]);
    return res;
}

FORCE_INLINE __m512i _mm512_packs_epi32 (__m512i a, __m512i b)
{
    __m512i res;
    res.vect_s16[0] = vcombine_s16(vqmovn_s32(a.vect_s32[0]), vqmovn_s32(b.vect_s32[0]));
    res.vect_s16[1] = vcombine_s16(vqmovn_s32(a.vect_s32[1]), vqmovn_s32(b.vect_s32[1]));
    res.vect_s16[2] = vcombine_s16(vqmovn_s32(a.vect_s32[2]), vqmovn_s32(b.vect_s32[2]));
    res.vect_s16[3] = vcombine_s16(vqmovn_s32(a.vect_s32[3]), vqmovn_s32(b.vect_s32[3]));
    return res;
}
