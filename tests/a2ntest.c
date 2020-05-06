/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
 * Description: avx2neon unit test
 * Author: xuqimeng
 * Create: 2019-11-05
 */

#include "a2ntest.h"

#include <math.h>
#include <string.h>

#include "avx2neontestdata.h"

const int g_256bit_divto_128bit = sizeof(__m256i) / sizeof(__m128i);
const int g_512bit_divto_128bit = sizeof(__m512i) / sizeof(__m128i);

const unsigned int M256I_M128I_NUM = 2U;
const unsigned int M256_M128_NUM = 2U;
const unsigned int M256D_M128D_NUM = 2U;

const unsigned int M512I_M128I_NUM = 4U;
const unsigned int M512_M128_NUM = 4U;
const unsigned int M512D_M128D_NUM = 4U;

const unsigned int M128I_INT8_NUM = 16U;
const unsigned int M128I_INT16_NUM = 8U;
const unsigned int M128I_INT32_NUM = 4U;
const unsigned int M128I_INT64_NUM = 2U;

const unsigned int M128_FLOAT32_NUM = 4U;
const unsigned int M128D_FLOAT64_NUM = 2U;

#define ASSERT_RETURN(x)                                                                                               \
    if (!(x))                                                                                                          \
        return FALSE;

#define MM256_CMP_PD(imm, rel, e)                                                                                      \
    do {                                                                                                               \
        for (i = 0; i < 4; i++)                                                                                        \
            e[i] = ((rel) != 0) ? -1 : 0;                                                                              \
        source1 = s1;                                                                                                  \
        source2 = s2;                                                                                                  \
        dest = _mm256_cmp_pd(source1, source2, imm);                                                                   \
        if (!comp_return(&dest, &e, 4 * sizeof(long long))) {                                                          \
            return FALSE;                                                                                              \
        }                                                                                                              \
    } while (0)

#define MM256_CMP_PS(imm, rel, e)                                                                                      \
    do {                                                                                                               \
        for (i = 0; i < 8; i++)                                                                                        \
            e[i] = ((rel) != 0) ? -1 : 0;                                                                              \
        source1 = s1;                                                                                                  \
        source2 = s2;                                                                                                  \
        dest = _mm256_cmp_ps(source1, source2, imm);                                                                   \
        if (!comp_return(&dest, &e, 8 * sizeof(int))) {                                                                \
            return FALSE;                                                                                              \
        }                                                                                                              \
    } while (0)

int comp_return(const void *src, const void *dst, const unsigned long len)
{
    return (0 == memcmp(src, dst, len) ? TRUE : FALSE);
}

const char *RunTest(InstructionTest test, int *flag)
{
    const char *ret = "UNKNOWN!";
    switch (test) {
        case UT_MM_POPCNT_U32:
            ret = "MM_POPCNT_U32";
            *flag = test_mm_popcnt_u32();
            break;
        case UT_MM_POPCNT_U64:
            ret = "MM_POPCNT_U64";
            *flag = test_mm_popcnt_u64();
            break;
        case UT_MM256_DIV_EPI16:
            ret = "MM256_DIV_EPI16";
            *flag = test_mm256_div_epi16();
            break;
        case UT_MM256_DIV_EPI32:
            ret = "MM256_DIV_EPI32";
            *flag = test_mm256_div_epi32();
            break;
        case UT_MM256_DIV_EPI64:
            ret = "MM256_DIV_EPI64";
            *flag = test_mm256_div_epi64();
            break;
        case UT_MM256_DIV_EPI8:
            ret = "MM256_DIV_EPI8";
            *flag = test_mm256_div_epi8();
            break;
        case UT_MM256_DIV_EPU16:
            ret = "MM256_DIV_EPU16";
            *flag = test_mm256_div_epu16();
            break;
        case UT_MM256_DIV_EPU32:
            ret = "MM256_DIV_EPU32";
            *flag = test_mm256_div_epu32();
            break;
        case UT_MM256_DIV_EPU64:
            ret = "MM256_DIV_EPU64";
            *flag = test_mm256_div_epu64();
            break;
        case UT_MM256_DIV_EPU8:
            ret = "MM256_DIV_EPU8";
            *flag = test_mm256_div_epu8();
            break;
        case UT_MM256_DIV_PD:
            ret = "MM256_DIV_PD";
            *flag = test_mm256_div_pd();
            break;
        case UT_MM256_DIV_PS:
            ret = "MM256_DIV_PS";
            *flag = test_mm256_div_ps();
            break;
        case UT_MM512_DIV_EPI16:
            ret = "MM512_DIV_EPI16";
            *flag = test_mm512_div_epi16();
            break;
        case UT_MM512_DIV_EPI32:
            ret = "MM512_DIV_EPI32";
            *flag = test_mm512_div_epi32();
            break;
        case UT_MM512_DIV_EPI64:
            ret = "MM512_DIV_EPI64";
            *flag = test_mm512_div_epi64();
            break;
        case UT_MM512_DIV_EPI8:
            ret = "MM512_DIV_EPI8";
            *flag = test_mm512_div_epi8();
            break;
        case UT_MM512_DIV_EPU16:
            ret = "MM512_DIV_EPU16";
            *flag = test_mm512_div_epu16();
            break;
        case UT_MM512_DIV_EPU32:
            ret = "MM512_DIV_EPU32";
            *flag = test_mm512_div_epu32();
            break;
        case UT_MM512_DIV_EPU64:
            ret = "MM512_DIV_EPU64";
            *flag = test_mm512_div_epu64();
            break;
        case UT_MM512_DIV_EPU8:
            ret = "MM512_DIV_EPU8";
            *flag = test_mm512_div_epu8();
            break;
        case UT_MM512_DIV_PD:
            ret = "MM512_DIV_PD";
            *flag = test_mm512_div_pd();
            break;
        case UT_MM512_DIV_PS:
            ret = "MM512_DIV_PS";
            *flag = test_mm512_div_ps();
            break;
        case UT_MM512_DIV_ROUND_PD:
            ret = "MM512_DIV_ROUND_PD";
            *flag = test_mm512_div_round_pd();
            break;
        case UT_MM512_DIV_ROUND_PS:
            ret = "MM512_DIV_ROUND_PS";
            *flag = test_mm512_div_round_ps();
            break;
        case UT_MM256_ADD_EPI8:
            ret = "MM256_ADD_EPI8";
            *flag = test_mm256_add_epi8();
            break;
        case UT_MM256_ADD_EPI16:
            ret = "MM256_ADD_EPI16";
            *flag = test_mm256_add_epi16();
            break;
        case UT_MM256_ADD_EPI32:
            ret = "MM256_ADD_EPI32";
            *flag = test_mm256_add_epi32();
            break;
        case UT_MM256_ADD_EPI64:
            ret = "MM256_ADD_EPI64";
            *flag = test_mm256_add_epi64();
            break;
        case UT_MM512_ADD_EPI8:
            ret = "MM512_ADD_EPI8";
            *flag = test_mm512_add_epi8();
            break;
        case UT_MM512_ADD_EPI16:
            ret = "MM512_ADD_EPI16";
            *flag = test_mm512_add_epi16();
            break;
        case UT_MM512_ADD_EPI32:
            ret = "MM512_ADD_EPI32";
            *flag = test_mm512_add_epi32();
            break;
        case UT_MM512_ADD_EPI64:
            ret = "MM512_ADD_EPI64";
            *flag = test_mm512_add_epi64();
            break;
        case UT_MM_ADDS_EPU8:
            ret = "MM_ADDS_EPU8";
            *flag = test_mm_adds_epu8();
            break;
        case UT_MM256_ADDS_EPI8:
            ret = "MM256_ADDS_EPI8";
            *flag = test_mm256_adds_epi8();
            break;
        case UT_MM256_ADDS_EPI16:
            ret = "MM256_ADDS_EPI16";
            *flag = test_mm256_adds_epi16();
            break;
        case UT_MM256_ADDS_EPU8:
            ret = "MM256_ADDS_EPU8";
            *flag = test_mm256_adds_epu8();
            break;
        case UT_MM256_ADDS_EPU16:
            ret = "MM256_ADDS_EPU16";
            *flag = test_mm256_adds_epu16();
            break;
        case UT_MM512_ADDS_EPI8:
            ret = "MM512_ADDS_EPI8";
            *flag = test_mm512_adds_epi8();
            break;
        case UT_MM512_ADDS_EPI16:
            ret = "MM512_ADDS_EPI16";
            *flag = test_mm512_adds_epi16();
            break;
        case UT_MM512_ADDS_EPU8:
            ret = "MM512_ADDS_EPU8";
            *flag = test_mm512_adds_epu8();
            break;
        case UT_MM512_ADDS_EPU16:
            ret = "MM512_ADDS_EPU16";
            *flag = test_mm512_adds_epu16();
            break;
        case UT_MM256_ADD_PS:
            ret = "MM256_ADD_PS";
            *flag = test_mm256_add_ps();
            break;
        case UT_MM256_ADD_PD:
            ret = "MM256_ADD_PD";
            *flag = test_mm256_add_pd();
            break;
        case UT_MM512_ADD_PS:
            ret = "MM512_ADD_PS";
            *flag = test_mm512_add_ps();
            break;
        case UT_MM512_ADD_PD:
            ret = "MM512_ADD_PD";
            *flag = test_mm512_add_pd();
            break;
        case UT_MM512_ADD_ROUND_PS:
            ret = "MM512_ADD_ROUND_PS";
            *flag = test_mm512_add_round_ps();
            break;
        case UT_MM512_ADD_ROUND_PD:
            ret = "MM512_ADD_ROUND_PD";
            *flag = test_mm512_add_round_pd();
            break;
        case UT_MM512_ADDN_PS:
            ret = "MM512_ADDN_PS";
            *flag = test_mm512_addn_ps();
            break;
        case UT_MM512_ADDN_PD:
            ret = "MM512_ADDN_PD";
            *flag = test_mm512_addn_pd();
            break;
        case UT_MM512_ADDN_ROUND_PS:
            ret = "MM512_ADDN_ROUND_PS";
            *flag = test_mm512_addn_round_ps();
            break;
        case UT_MM512_ADDN_ROUND_PD:
            ret = "MM512_ADDN_ROUND_PD";
            *flag = test_mm512_addn_round_pd();
            break;
        case UT_MM512_ADDSETC_EPI32:
            ret = "MM512_ADDSETC_EPI32";
            *flag = test_mm512_addsetc_epi32();
            break;
        case UT_MM512_ADDSETS_EPI32:
            ret = "MM512_ADDSETS_EPI32";
            *flag = test_mm512_addsets_epi32();
            break;
        case UT_MM512_ADDSETS_PS:
            ret = "MM512_ADDSETS_PS";
            *flag = test_mm512_addsets_ps();
            break;
        case UT_MM512_ADDSETS_ROUND_PS:
            ret = "MM512_ADDSETS_ROUND_PS";
            *flag = test_mm512_addsets_round_ps();
            break;
        case UT_MM256_ADDSUB_PS:
            ret = "MM256_ADDSUB_PS";
            *flag = test_mm256_addsub_ps();
            break;
        case UT_MM256_ADDSUB_PD:
            ret = "MM256_ADDSUB_PD";
            *flag = test_mm256_addsub_pd();
            break;
        case UT_MM_SUB_EPI8:
            ret = "MM_SUB_EPI8";
            *flag = test_mm_sub_epi8();
            break;
        case UT_MM256_SUB_EPI16:
            ret = "MM256_SUB_EPI16";
            *flag = test_mm256_sub_epi16();
            break;
        case UT_MM256_SUB_EPI32:
            ret = "MM256_SUB_EPI32";
            *flag = test_mm256_sub_epi32();
            break;
        case UT_MM256_SUB_EPI64:
            ret = "MM256_SUB_EPI64";
            *flag = test_mm256_sub_epi64();
            break;
        case UT_MM256_SUB_EPI8:
            ret = "MM256_SUB_EPI8";
            *flag = test_mm256_sub_epi8();
            break;
        case UT_MM256_SUB_PD:
            ret = "MM256_SUB_PD";
            *flag = test_mm256_sub_pd();
            break;
        case UT_MM256_SUB_PS:
            ret = "MM256_SUB_PS";
            *flag = test_mm256_sub_ps();
            break;
        case UT_MM512_SUB_EPI16:
            ret = "MM512_SUB_EPI16";
            *flag = test_mm512_sub_epi16();
            break;
        case UT_MM512_SUB_EPI32:
            ret = "MM512_SUB_EPI32";
            *flag = test_mm512_sub_epi32();
            break;
        case UT_MM512_SUB_EPI64:
            ret = "MM512_SUB_EPI64";
            *flag = test_mm512_sub_epi64();
            break;
        case UT_MM512_SUB_EPI8:
            ret = "MM512_SUB_EPI8";
            *flag = test_mm512_sub_epi8();
            break;
        case UT_MM512_SUB_PD:
            ret = "MM512_SUB_PD";
            *flag = test_mm512_sub_pd();
            break;
        case UT_MM512_SUB_PS:
            ret = "MM512_SUB_PS";
            *flag = test_mm512_sub_ps();
            break;
        case UT_MM256_SUBS_EPI16:
            ret = "MM256_SUBS_EPI16";
            *flag = test_mm256_subs_epi16();
            break;
        case UT_MM256_SUBS_EPI8:
            ret = "MM256_SUBS_EPI8";
            *flag = test_mm256_subs_epi8();
            break;
        case UT_MM256_SUBS_EPU16:
            ret = "MM256_SUBS_EPU16";
            *flag = test_mm256_subs_epu16();
            break;
        case UT_MM256_SUBS_EPU8:
            ret = "MM256_SUBS_EPU8";
            *flag = test_mm256_subs_epu8();
            break;
        case UT_MM512_SUBS_EPI16:
            ret = "MM512_SUBS_EPI16";
            *flag = test_mm512_subs_epi16();
            break;
        case UT_MM512_SUBS_EPI8:
            ret = "MM512_SUBS_EPI8";
            *flag = test_mm512_subs_epi8();
            break;
        case UT_MM512_SUBS_EPU16:
            ret = "MM512_SUBS_EPU16";
            *flag = test_mm512_subs_epu16();
            break;
        case UT_MM512_SUBS_EPU8:
            ret = "MM512_SUBS_EPU8";
            *flag = test_mm512_subs_epu8();
            break;
        case UT_MM512_SUB_ROUND_PD:
            ret = "MM512_SUB_ROUND_PD";
            *flag = test_mm512_sub_round_pd();
            break;
        case UT_MM512_SUB_ROUND_PS:
            ret = "MM512_SUB_ROUND_PS";
            *flag = test_mm512_sub_round_ps();
            break;
        case UT_MM512_SUBR_EPI32:
            ret = "MM512_SUBR_EPI32";
            *flag = test_mm512_subr_epi32();
            break;
        case UT_MM512_SUBR_PS:
            ret = "MM512_SUBR_PS";
            *flag = test_mm512_subr_ps();
            break;
        case UT_MM512_SUBR_PD:
            ret = "MM512_SUBR_PD";
            *flag = test_mm512_subr_pd();
            break;
        case UT_MM512_SUBR_ROUND_PS:
            ret = "MM512_SUBR_ROUND_PS";
            *flag = test_mm512_subr_round_ps();
            break;
        case UT_MM512_SUBR_ROUND_PD:
            ret = "MM512_SUBR_ROUND_PD";
            *flag = test_mm512_subr_round_pd();
            break;
        case UT_MM512_SUBSETB_EPI32:
            ret = "MM512_SUBSETB_EPI32";
            *flag = test_mm512_subsetb_epi32();
            break;
        case UT_MM512_SUBRSETB_EPI32:
            ret = "MM512_SUBRSETB_EPI32";
            *flag = test_mm512_subrsetb_epi32();
            break;
        case UT_MM256_ZEROUPPER:
            ret = "MM256_ZEROUPPER";
            *flag = test_mm256_zeroupper();
            break;
        case UT_MM512_BSLLI_EPI128:
            ret = "MM512_BSLLI_EPI128";
            *flag = test_mm512_bslli_epi128();
            break;
        case UT_MM512_BSRLI_EPI128:
            ret = "MM512_BSRLI_EPI128";
            *flag = test_mm512_bsrli_epi128();
            break;
        case UT_MM512_PERMUTEXVAR_EPI64:
            ret = "MM512_PERMUTEXVAR_EPI64";
            *flag = test_mm512_permutexvar_epi64();
            break;
        case UT_MM512_EXTRACTI32X4_EPI32:
            ret = "MM512_EXTRACTI32X4_EPI32";
            *flag = test_mm512_extracti32x4_epi32();
            break;
        case UT_MM512_TEST_EPI8_MASK:
            ret = "MM512_TEST_EPI8_MASK";
            *flag = test_mm512_test_epi8_mask();
            break;
        case UT_MM512_TEST_EPI32_MASK:
            ret = "MM512_TEST_EPI32_MASK";
            *flag = test_mm512_test_epi32_mask();
            break;
        case UT_MM512_TEST_EPI64_MASK:
            ret = "MM512_TEST_EPI64_MASK";
            *flag = test_mm512_test_epi64_mask();
            break;
        case UT_MM256_MUL_EPI32:
            ret = "MM256_MUL_EPI32";
            *flag = test_mm256_mul_epi32();
            break;
        case UT_MM256_MUL_EPU32:
            ret = "MM256_MUL_EPU32";
            *flag = test_mm256_mul_epu32();
            break;
        case UT_MM256_MUL_PD:
            ret = "MM256_MUL_PD";
            *flag = test_mm256_mul_pd();
            break;
        case UT_MM256_MUL_PS:
            ret = "MM256_MUL_PS";
            *flag = test_mm256_mul_ps();
            break;
        case UT_MM512_MUL_EPI32:
            ret = "MM512_MUL_EPI32";
            *flag = test_mm512_mul_epi32();
            break;
        case UT_MM512_MUL_EPU32:
            ret = "MM512_MUL_EPU32";
            *flag = test_mm512_mul_epu32();
            break;
        case UT_MM512_MUL_PD:
            ret = "MM512_MUL_PD";
            *flag = test_mm512_mul_pd();
            break;
        case UT_MM512_MUL_PS:
            ret = "MM512_MUL_PS";
            *flag = test_mm512_mul_ps();
            break;
        case UT_MM256_MULHI_EPI16:
            ret = "MM256_MULHI_EPI16";
            *flag = test_mm256_mulhi_epi16();
            break;
        case UT_MM256_MULHI_EPU16:
            ret = "MM256_MULHI_EPU16";
            *flag = test_mm256_mulhi_epu16();
            break;
        case UT_MM512_MULHI_EPI16:
            ret = "MM512_MULHI_EPI16";
            *flag = test_mm512_mulhi_epi16();
            break;
        case UT_MM512_MULHI_EPU16:
            ret = "MM512_MULHI_EPU16";
            *flag = test_mm512_mulhi_epu16();
            break;
        case UT_MM512_MULHI_EPI32:
            ret = "MM512_MULHI_EPI32";
            *flag = test_mm512_mulhi_epi32();
            break;
        case UT_MM512_MULHI_EPU32:
            ret = "MM512_MULHI_EPU32";
            *flag = test_mm512_mulhi_epu32();
            break;
        case UT_MM256_MULLO_EPI16:
            ret = "MM256_MULLO_EPI16";
            *flag = test_mm256_mullo_epi16();
            break;
        case UT_MM256_MULLO_EPI32:
            ret = "MM256_MULLO_EPI32";
            *flag = test_mm256_mullo_epi32();
            break;
        case UT_MM256_MULLO_EPI64:
            ret = "MM256_MULLO_EPI64";
            *flag = test_mm256_mullo_epi64();
            break;
        case UT_MM512_MULLO_EPI16:
            ret = "MM512_MULLO_EPI16";
            *flag = test_mm512_mullo_epi16();
            break;
        case UT_MM512_MULLO_EPI32:
            ret = "MM512_MULLO_EPI32";
            *flag = test_mm512_mullo_epi32();
            break;
        case UT_MM512_MULLO_EPI64:
            ret = "MM512_MULLO_EPI64";
            *flag = test_mm512_mullo_epi64();
            break;
        case UT_MM512_MULLOX_EPI64:
            ret = "MM512_MULLOX_EPI64";
            *flag = test_mm512_mullox_epi64();
            break;
        case UT_MM256_MULHRS_EPI16:
            ret = "MM256_MULHRS_EPI16";
            *flag = test_mm256_mulhrs_epi16();
            break;
        case UT_MM512_MULHRS_EPI16:
            ret = "MM512_MULHRS_EPI16";
            *flag = test_mm512_mulhrs_epi16();
            break;
        case UT_MM512_MUL_ROUND_PD:
            ret = "MM512_MUL_ROUND_PD";
            *flag = test_mm512_mul_round_pd();
            break;
        case UT_MM512_MUL_ROUND_PS:
            ret = "MM512_MUL_ROUND_PS";
            *flag = test_mm512_mul_round_ps();
            break;
        case UT_MM_SLL_EPI64:
            ret = "MM_SLL_EPI64";
            *flag = test_mm_sll_epi64();
            break;
        case UT_MM_SLLI_SI128:
            ret = "MM_SLLI_SI128";
            *flag = test_mm_slli_si128();
            break;
        case UT_MM_SRLI_SI128:
            ret = "MM_SRLI_SI128";
            *flag = test_mm_srli_si128();
            break;
        case UT_MM_SLLI_EPI32:
            ret = "MM_SLLI_EPI32";
            *flag = test_mm_slli_epi32();
            break;
        case UT_MM_SLLI_EPI64:
            ret = "MM_SLLI_EPI64";
            *flag = test_mm_slli_epi64();
            break;
        case UT_MM_SRLI_EPI64:
            ret = "MM_SRLI_EPI64";
            *flag = test_mm_srli_epi64();
            break;
        case UT_MM256_SLL_EPI32:
            ret = "MM256_SLL_EPI32";
            *flag = test_mm256_sll_epi32();
            break;
        case UT_MM256_SLL_EPI64:
            ret = "MM256_SLL_EPI64";
            *flag = test_mm256_sll_epi64();
            break;
        case UT_MM256_SLLI_EPI64:
            ret = "MM256_SLLI_EPI64";
            *flag = test_mm256_slli_epi64();
            break;
        case UT_MM256_SLLI_EPI32:
            ret = "MM256_SLLI_EPI32";
            *flag = test_mm256_slli_epi32();
            break;
        case UT_MM256_SRLI_EPI64:
            ret = "MM256_SRLI_EPI64";
            *flag = test_mm256_srli_epi64();
            break;
        case UT_MM512_SLL_EPI64:
            ret = "MM512_SLL_EPI64";
            *flag = test_mm512_sll_epi64();
            break;
        case UT_MM512_SLLI_EPI64:
            ret = "MM512_SLLI_EPI64";
            *flag = test_mm512_slli_epi64();
            break;
        case UT_MM512_SRLI_EPI64:
            ret = "MM512_SRLI_EPI64";
            *flag = test_mm512_srli_epi64();
            break;
        case UT_MM256_SLLI_SI256:
            ret = "MM256_SLLI_SI256";
            *flag = test_mm256_slli_si256();
            break;
        case UT_MM256_SRLI_SI256:
            ret = "MM256_SRLI_SI256";
            *flag = test_mm256_srli_si256();
            break;
        case UT_MM256_BLENDV_PD:
            ret = "MM256_BLENDV_PD";
            *flag = test_mm256_blendv_pd();
            break;
        case UT_MM256_BLENDV_PS:
            ret = "MM256_BLENDV_PS";
            *flag = test_mm256_blendv_ps();
            break;
        case UT_MM256_BLEND_PD:
            ret = "MM256_BLEND_PD";
            *flag = test_mm256_blend_pd();
            break;
        case UT_MM256_BLEND_PS:
            ret = "MM256_BLEND_PS";
            *flag = test_mm256_blend_ps();
            break;
        case UT_MM512_MASK_BLEND_EPI32:
            ret = "MM512_MASK_BLEND_EPI32";
            *flag = test_mm512_mask_blend_epi32();
            break;
        case UT_MM512_MASK_BLEND_PD:
            ret = "MM512_MASK_BLEND_PD";
            *flag = test_mm512_mask_blend_pd();
            break;
        case UT_MM512_MASK_BLEND_PS:
            ret = "MM512_MASK_BLEND_PS";
            *flag = test_mm512_mask_blend_ps();
            break;
        case UT_MM_AND_SI128:
            ret = "MM_AND_SI128";
            *flag = test_mm_and_si128();
            break;
        case UT_MM256_AND_SI256:
            ret = "MM256_AND_SI256";
            *flag = test_mm256_and_si256();
            break;
        case UT_MM512_AND_SI512:
            ret = "MM512_AND_SI512";
            *flag = test_mm512_and_si512();
            break;
        case UT_MM_OR_SI128:
            ret = "MM_OR_SI128";
            *flag = test_mm_or_si128();
            break;
        case UT_MM256_OR_SI256:
            ret = "MM256_OR_SI256";
            *flag = test_mm256_or_si256();
            break;
        case UT_MM512_OR_SI512:
            ret = "MM512_OR_SI512";
            *flag = test_mm512_or_si512();
            break;
        case UT_MM_ANDNOT_SI128:
            ret = "MM_ANDNOT_SI128";
            *flag = test_mm_andnot_si128();
            break;
        case UT_MM256_ANDNOT_SI256:
            ret = "MM256_ANDNOT_SI256";
            *flag = test_mm256_andnot_si256();
            break;
        case UT_MM512_ANDNOT_SI512:
            ret = "MM512_ANDNOT_SI512";
            *flag = test_mm512_andnot_si512();
            break;
        case UT_MM_XOR_SI128:
            ret = "MM_XOR_SI128";
            *flag = test_mm_xor_si128();
            break;
        case UT_MM256_XOR_SI256:
            ret = "MM256_XOR_SI256";
            *flag = test_mm256_xor_si256();
            break;
        case UT_MM512_XOR_SI512:
            ret = "MM512_XOR_SI512";
            *flag = test_mm512_xor_si512();
            break;
        case UT_MM256_OR_PS:
            ret = "MM256_OR_PS";
            *flag = test_mm256_or_ps();
            break;
        case UT_MM256_OR_PD:
            ret = "MM256_OR_PS";
            *flag = test_mm256_or_pd();
            break;
        case UT_MM512_AND_EPI32:
            ret = "MM512_AND_EPI32";
            *flag = test_mm512_and_epi32();
            break;
        case UT_MM512_AND_EPI64:
            ret = "MM512_AND_EPI64";
            *flag = test_mm512_and_epi64();
            break;
        case UT_MM512_OR_EPI32:
            ret = "MM512_OR_EPI32";
            *flag = test_mm512_or_epi32();
            break;
        case UT_MM512_OR_EPI64:
            ret = "MM512_OR_EPI64";
            *flag = test_mm512_or_epi64();
            break;
        case UT_MM512_XOR_PS:
            ret = "MM512_XOR_PS";
            *flag = test_mm512_xor_ps();
            break;
        case UT_MM512_XOR_PD:
            ret = "MM512_XOR_PD";
            *flag = test_mm512_xor_pd();
            break;
        case UT_MM_CMPEQ_EPI8:
            ret = "MM_CMPEQ_EPI8";
            *flag = test_mm_cmpeq_epi8();
            break;
        case UT_MM_CMPEQ_EPI32:
            ret = "MM_CMPEQ_EPI32";
            *flag = test_mm_cmpeq_epi32();
            break;
        case UT_MM256_CMPGT_EPI32:
            ret = "MM256_CMPGT_EPI32";
            *flag = test_mm256_cmpgt_epi32();
            break;
        case UT_MM256_CMPEQ_EPI8:
            ret = "MM256_CMPEQ_EPI8";
            *flag = test_mm256_cmpeq_epi8();
            break;
        case UT_MM256_CMPEQ_EPI32:
            ret = "MM256_CMPEQ_EPI32";
            *flag = test_mm256_cmpeq_epi32();
            break;
        case UT_MM_CMPEQ_EPI64:
            ret = "MM_CMPEQ_EPI64";
            *flag = test_mm_cmpeq_epi64();
            break;
        case UT_MM512_CMP_EPI32_MASK:
            ret = "MM512_CMP_EPI32_MASK";
            *flag = test_mm512_cmp_epi32_mask();
            break;
        case UT_MM512_CMP_EPI8_MASK:
            ret = "MM512_CMP_EPI8_MASK";
            *flag = test_mm512_cmp_epi8_mask();
            break;
        case UT_MM512_CMPEQ_EPI8_MASK:
            ret = "MM512_CMPEQ_EPI8_MASK";
            *flag = test_mm512_cmpeq_epi8_mask();
            break;
        case UT_MM512_CMPGT_EPI32_MASK:
            ret = "MM512_CMPGT_EPI32_MASK";
            *flag = test_mm512_cmpgt_epi32_mask();
            break;
        case UT_MM512_MASK_CMPEQ_EPI8_MASK:
            ret = "MM512_MASK_CMPEQ_EPI8_MASK";
            *flag = test_mm512_mask_cmpeq_epi8_mask();
            break;
        case UT_MM512_SET_EPI32:
            ret = "MM512_SET_EPI32";
            *flag = test_mm512_set_epi32();
            break;
        case UT_MM512_SET_EPI64:
            ret = "MM512_SET_EPI64";
            *flag = test_mm512_set_epi64();
            break;
        case UT_MM512_SET1_EPI32:
            ret = "MM512_SET1_EPI32";
            *flag = test_mm512_set1_epi32();
            break;
        case UT_MM512_SET1_EPI64:
            ret = "MM512_SET1_EPI64";
            *flag = test_mm512_set1_epi64();
            break;
        case UT_MM512_SET1_EPI8:
            ret = "MM512_SET1_EPI8";
            *flag = test_mm512_set1_epi8();
            break;
        case UT_MM512_SET_PS:
            ret = "MM512_SET_PS";
            *flag = test_mm512_set_ps();
            break;
        case UT_MM512_SET_PD:
            ret = "MM512_SET_PD";
            *flag = test_mm512_set_pd();
            break;
        case UT_MM512_SET1_PS:
            ret = "MM512_SET1_PS";
            *flag = test_mm512_set1_ps();
            break;
        case UT_MM512_SET1_PD:
            ret = "MM512_SET1_PD";
            *flag = test_mm512_set1_pd();
            break;
        case UT_MM512_SETZERO_PS:
            ret = "MM512_SETZERO_PS";
            *flag = test_mm512_setzero_ps();
            break;
        case UT_MM512_SETZERO_PD:
            ret = "MM512_SETZERO_PD";
            *flag = test_mm512_setzero_pd();
            break;
        case UT_MM_MOVE_SD:
            ret = "MM_MOVE_SD";
            *flag = test_mm_move_sd();
            break;
        case UT_MM_MOVE_SS:
            ret = "MM_MOVE_SS";
            *flag = test_mm_move_ss();
            break;
        case UT_MM_MOVEMASK_EPI8:
            ret = "MM_MOVEMASK_EPI8";
            *flag = test_mm_movemask_epi8();
            break;
        case UT_MM_MOVEMASK_PS:
            ret = "MM_MOVEMASK_PS";
            *flag = test_mm_movemask_ps();
            break;
        case UT_MM256_MOVEMASK_EPI8:
            ret = "MM256_MOVEMASK_EPI8";
            *flag = test_mm256_movemask_epi8();
            break;
        case UT_MM256_MOVEMASK_PS:
            ret = "MM256_MOVEMASK_PS";
            *flag = test_mm256_movemask_ps();
            break;
        case UT_MM_TESTZ_SI128:
            ret = "MM_TESTZ_SI128";
            *flag = test_mm_testz_si128();
            break;
        case UT_MM256_TESTZ_SI256:
            ret = "MM256_TESTZ_SI256";
            *flag = test_mm256_testz_si256();
            break;
        case UT_MM512_MOVM_EPI8:
            ret = "MM512_MOVM_EPI8";
            *flag = test_mm512_movm_epi8();
            break;
        case UT_MM512_MOVM_EPI32:
            ret = "MM512_MOVM_EPI32";
            *flag = test_mm512_movm_epi32();
            break;
        case UT_MM_EXTRACT_EPI32:
            ret = "MM_EXTRACT_EPI32";
            *flag = test_mm_extract_epi32();
            break;
        case UT_MM_EXTRACT_EPI64:
            ret = "MM_EXTRACT_EPI64";
            *flag = test_mm_extract_epi64();
            break;
        case UT_MM256_EXTRACTI128_SI256:
            ret = "MM256_EXTRACTI128_SI256";
            *flag = test_mm256_extracti128_si256();
            break;
        case UT_MM_EXTRACT_PS:
            ret = "MM_EXTRACT_PS";
            *flag = test_mm_extract_ps();
            break;
        case UT_MM256_EXTRACT_EPI32:
            ret = "MM256_EXTRACT_EPI32";
            *flag = test_mm256_extract_epi32();
            break;
        case UT_MM256_EXTRACT_EPI64:
            ret = "MM256_EXTRACT_EPI64";
            *flag = test_mm256_extract_epi64();
            break;
        case UT_MM256_EXTRACTF128_PS:
            ret = "MM256_EXTRACTF128_PS";
            *flag = test_mm256_extractf128_ps();
            break;
        case UT_MM256_EXTRACTF128_PD:
            ret = "MM256_EXTRACTF128_PD";
            *flag = test_mm256_extractf128_pd();
            break;
        case UT_MM512_EXTRACTF32x8_PS:
            ret = "MM512_EXTRACTF32x8_PS";
            *flag = test_mm512_extractf32x8_ps();
            break;
        case UT_MM512_EXTRACTF64x4_PD:
            ret = "MM512_EXTRACTF64x4_PD";
            *flag = test_mm512_extractf64x4_pd();
            break;
        case UT_MM_CRC32_U8:
            ret = "MM_CRC32_U8";
            *flag = test_mm_crc32_u8();
            break;
        case UT_MM_CRC32_U16:
            ret = "MM_CRC32_U16";
            *flag = test_mm_crc32_u16();
            break;
        case UT_MM_CRC32_U32:
            ret = "MM_CRC32_U32";
            *flag = test_mm_crc32_u32();
            break;
        case UT_MM_CRC32_U64:
            ret = "MM_CRC32_U64";
            *flag = test_mm_crc32_u64();
            break;
        case UT_MM256_UNPACKLO_EPI8:
            ret = "MM256_UNPACKLO_EPI8";
            *flag = test_mm256_unpacklo_epi8();
            break;
        case UT_MM256_UNPACKHI_EPI8:
            ret = "MM256_UNPACKHI_EPI8";
            *flag = test_mm256_unpackhi_epi8();
            break;
        case UT_MM512_UNPACKLO_EPI8:
            ret = "MM512_UNPACKLO_EPI8";
            *flag = test_mm512_unpacklo_epi8();
            break;
        case UT_MM512_UNPACKHI_EPI8:
            ret = "MM512_UNPACKHI_EPI8";
            *flag = test_mm512_unpackhi_epi8();
            break;

        case UT_MM_STOREU_SI128:
            ret = "MM_STOREU_SI128";
            *flag = test_mm_storeu_si128();
            break;
        case UT_MM256_STORE_SI256:
            ret = "MM256_STORE_SI256";
            *flag = test_mm256_store_si256();
            break;
        case UT_MM256_STOREU_SI256:
            ret = "MM256_STOREU_SI256";
            *flag = test_mm256_storeu_si256();
            break;
        case UT_MM256_STREAM_SI256:
            ret = "MM256_STREAM_SI256";
            *flag = test_mm256_stream_si256();
            break;
        case UT_MM512_STORE_SI512:
            ret = "MM512_STORE_SI512";
            *flag = test_mm512_store_si512();
            break;
        case UT_MM512_STOREU_SI512:
            ret = "MM512_STOREU_SI512";
            *flag = test_mm512_storeu_si512();
            break;
        case UT_MM512_STREAM_SI512:
            ret = "MM512_STREAM_SI512";
            *flag = test_mm512_stream_si512();
            break;
        case UT_MM256_INSERTI128_SI256:
            ret = "MM256_INSERTI128_SI256";
            *flag = test_mm256_inserti128_si256();
            break;
        case UT_MM256_INSERTF128_PD:
            ret = "MM256_INSERTF128_PD";
            *flag = test_mm256_insertf128_pd();
            break;
        case UT_MM256_INSERTF128_PS:
            ret = "MM256_INSERTF128_PS";
            *flag = test_mm256_insertf128_ps();
            break;
        case UT_MM256_PERMUTE4X64_EPI64:
            ret = "MM256_PERMUTE4X64_EPI64";
            *flag = test_mm256_permute4x64_epi64();
            break;
        case UT_MM256_PERMUTE2F128_SI256:
            ret = "MM256_PERMUTE2F128_SI256";
            *flag = test_mm256_permute2f128_si256();
            break;
        case UT_MM_SET_PD:
            ret = "MM_SET_PD";
            *flag = test_mm_set_pd();
            break;
        case UT_MM256_SET_EPI32:
            ret = "MM256_SET_EPI32";
            *flag = test_mm256_set_epi32();
            break;
        case UT_MM256_SET_EPI64X:
            ret = "MM256_SET_EPI64X";
            *flag = test_mm256_set_epi64x();
            break;
        case UT_MM256_SET_M128I:
            ret = "MM256_SET_M128I";
            *flag = test_mm256_set_m128i();
            break;
        case UT_MM256_SET_PS:
            ret = "MM256_SET_PS";
            *flag = test_mm256_set_ps();
            break;
        case UT_MM256_SET_PD:
            ret = "MM256_SET_PD";
            *flag = test_mm256_set_pd();
            break;
        case UT_MM_SETZERO_SI128:
            ret = "MM_SETZERO_SI128";
            *flag = test_mm_setzero_si128();
            break;
        case UT_MM256_SETZERO_SI256:
            ret = "MM256_SETZERO_SI256";
            *flag = test_mm256_setzero_si256();
            break;
        case UT_MM512_SETZERO_SI512:
            ret = "MM512_SETZERO_SI512";
            *flag = test_mm512_setzero_si512();
            break;
        case UT_MM256_SETZERO_PS:
            ret = "MM256_SETZERO_PS";
            *flag = test_mm256_setzero_ps();
            break;
        case UT_MM256_SETZERO_PD:
            ret = "MM256_SETZERO_PD";
            *flag = test_mm256_setzero_pd();
            break;
        case UT_MM_SET1_EPI8:
            ret = "MM_SET1_EPI8";
            *flag = test_mm_set1_epi8();
            break;
        case UT_MM_SET1_EPI32:
            ret = "MM_SET1_EPI32";
            *flag = test_mm_set1_epi32();
            break;
        case UT_MM_SET1_PS:
            ret = "MM_SET1_PS";
            *flag = test_mm_set1_ps();
            break;
        case UT_MM_SET1_EPI64X:
            ret = "MM_SET1_EPI64X";
            *flag = test_mm_set1_epi64x();
            break;
        case UT_MM_SET1_PD:
            ret = "MM_SET1_PD";
            *flag = test_mm_set1_pd();
            break;
        case UT_MM256_SET1_EPI8:
            ret = "MM256_SET1_EPI8";
            *flag = test_mm256_set1_epi8();
            break;
        case UT_MM256_SET1_EPI32:
            ret = "MM256_SET1_EPI32";
            *flag = test_mm256_set1_epi32();
            break;
        case UT_MM256_SET1_EPI64X:
            ret = "MM256_SET1_EPI64X";
            *flag = test_mm256_set1_epi64x();
            break;
        case UT_MM256_SET1_PD:
            ret = "MM256_SET1_PD";
            *flag = test_mm256_set1_pd();
            break;
        case UT_MM256_SET1_PS:
            ret = "MM256_SET1_PS";
            *flag = test_mm256_set1_ps();
            break;
        case UT_MM_LOADU_SI128:
            ret = "MM_LOADU_SI128";
            *flag = test_mm_loadu_si128();
            break;
        case UT_MM256_LOAD_SI256:
            ret = "MM256_LOAD_SI256";
            *flag = test_mm256_load_si256();
            break;
        case UT_MM256_LOADU_SI256:
            ret = "MM256_LOADU_SI256";
            *flag = test_mm256_loadu_si256();
            break;
        case UT_MM256_MASKLOAD_EPI32:
            ret = "MM256_MASKLOAD_EPI32";
            *flag = test_mm256_maskload_epi32();
            break;
        case UT_MM512_LOAD_SI512:
            ret = "MM512_LOAD_SI512";
            *flag = test_mm512_load_si512();
            break;
        case UT_MM512_LOADU_SI512:
            ret = "MM512_LOADU_SI512";
            *flag = test_mm512_loadu_si512();
            break;
        case UT_MM512_MASK_LOADU_EPI8:
            ret = "MM512_MASK_LOADU_EPI8";
            *flag = test_mm512_mask_loadu_epi8();
            break;
        case UT_MM512_MASKZ_LOADU_EPI8:
            ret = "MM512_MASKZ_LOADU_EPI8";
            *flag = test_mm512_maskz_loadu_epi8();
            break;
        case UT_MM512_ABS_EPI8:
            ret = "MM512_ABS_EPI8";
            *flag = test_mm512_abs_epi8();
            break;
        case UT_MM256_BROADCASTQ_EPI64:
            ret = "MM256_BROADCASTQ_EPI64";
            *flag = test_mm256_broadcastq_epi64();
            break;
        case UT_MM256_BROADCASTSI128_SI256:
            ret = "MM256_BROADCASTSI128_SI256";
            *flag = test_mm256_broadcastsi128_si256();
            break;
        case UT_MM512_BROADCAST_I32X4:
            ret = "MM512_BROADCAST_I32X4";
            *flag = test_mm512_broadcast_i32x4();
            break;
        case UT_MM512_BROADCAST_I64X4:
            ret = "MM512_BROADCAST_I64X4";
            *flag = test_mm512_broadcast_i64x4();
            break;
        case UT_MM512_MASK_BROADCAST_I64X4:
            ret = "MM512_MASK_BROADCAST_I64X4";
            *flag = test_mm512_mask_broadcast_i64x4();
            break;
        case UT_MM256_CASTPD128_PD256:
            ret = "MM256_CASTPD128_PD256";
            *flag = test_mm256_castpd128_pd256();
            break;
        case UT_MM256_CASTPD256_PD128:
            ret = "MM256_CASTPD256_PD128";
            *flag = test_mm256_castpd256_pd128();
            break;
        case UT_MM256_CASTPS128_PS256:
            ret = "MM256_CASTPS128_PS256";
            *flag = test_mm256_castps128_ps256();
            break;
        case UT_MM256_CASTPS256_PS128:
            ret = "MM256_CASTPS256_PS128";
            *flag = test_mm256_castps256_ps128();
            break;
        case UT_MM_CASTSI128_PS:
            ret = "MM_CASTSI128_PS";
            *flag = test_mm_castsi128_ps();
            break;
        case UT_MM256_CASTSI128_SI256:
            ret = "MM256_CASTSI128_SI256";
            *flag = test_mm256_castsi128_si256();
            break;
        case UT_MM256_CASTSI256_PS:
            ret = "MM256_CASTSI256_PS";
            *flag = test_mm256_castsi256_ps();
            break;
        case UT_MM256_CASTSI256_SI128:
            ret = "MM256_CASTSI256_SI128";
            *flag = test_mm256_castsi256_si128();
            break;
        case UT_MM_CVTSI32_SI128:
            ret = "MM_CVTSI32_SI128";
            *flag = test_mm_cvtsi32_si128();
            break;
        case UT_MM_CVTSI128_SI32:
            ret = "MM_CVTSI128_SI32";
            *flag = test_mm_cvtsi128_si32();
            break;
        case UT_MM256_CVTEPI32_PD:
            ret = "MM256_CVTEPI32_PD";
            *flag = test_mm256_cvtepi32_pd();
            break;
        case UT_MM256_CVTEPI32_PS:
            ret = "MM256_CVTEPI32_PS";
            *flag = test_mm256_cvtepi32_ps();
            break;
        case UT_MM_SHUFFLE_EPI8:
            ret = "MM_SHUFFLE_EPI8";
            *flag = test_mm_shuffle_epi8();
            break;
        case UT_MM256_SHUFFLE_EPI8:
            ret = "MM256_SHUFFLE_EPI8";
            *flag = test_mm256_shuffle_epi8();
            break;
        case UT_MM512_SHUFFLE_EPI8:
            ret = "MM512_SHUFFLE_EPI8";
            *flag = test_mm512_shuffle_epi8();
            break;
        case UT_MM512_MASKZ_SHUFFLE_EPI8:
            ret = "MM512_MASKZ_SHUFFLE_EPI8";
            *flag = test_mm512_maskz_shuffle_epi8();
            break;
        case UT_MM256_MULTISHIFT_EPI64_EPI8:
            ret = "MM256_MULTISHIFT_EPI64_EPI8";
            *flag = test_mm256_multishift_epi64_epi8();
            break;
        case UT_MM512_MULTISHIFT_EPI64_EPI8:
            ret = "MM512_MULTISHIFT_EPI64_EPI8";
            *flag = test_mm512_multishift_epi64_epi8();
            break;
        case UT_MM256_ALIGNR_EPI8:
            ret = "MM256_ALIGNR_EPI8";
            *flag = test_mm256_alignr_epi8();
            break;
        case UT_MM_CMPESTRI:
            ret = "MM_CMPESTRI";
            *flag = test_mm_cmpestri();
            break;
        case UT_MM_CMPESTRM:
            ret = "MM_CMPESTRM";
            *flag = test_mm_cmpestrm();
            break;
        case UT_MM_INSERT_EPI32:
            ret = "MM_INSERT_EPI32";
            *flag = test_mm_insert_epi32();
            break;
        case UT_MM256_INSERT_EPI32:
            ret = "MM256_INSERT_EPI32";
            *flag = test_mm256_insert_epi32();
            break;
        case UT_MM256_INSERT_EPI64:
            ret = "MM256_INSERT_EPI64";
            *flag = test_mm256_insert_epi64();
            break;
        case UT_MM512_CASTPD128_PD512:
            ret = "MM512_CASTPD128_PD512";
            *flag = test_mm512_castpd128_pd512();
            break;
        case UT_MM512_CASTPD512_PD128:
            ret = "MM512_CASTPD512_PD128";
            *flag = test_mm512_castpd512_pd128();
            break;
        case UT_MM512_CASTPS128_PS512:
            ret = "MM512_CASTPS128_PS512";
            *flag = test_mm512_castps128_ps512();
            break;
        case UT_MM512_CASTPS512_PS128:
            ret = "MM512_CASTPS512_PS128";
            *flag = test_mm512_castps512_ps128();
            break;
        case UT_MM512_CVTEPI32_PD:
            ret = "MM512_CVTEPI32_PD";
            *flag = test_mm512_cvtepi32_pd();
            break;
        case UT_MM512_CVTEPI32_PS:
            ret = "MM512_CVTEPI32_PS";
            *flag = test_mm512_cvtepi32_ps();
            break;
        case UT_MM512_INSERTF32X8:
            ret = "MM512_INSERTF32X8";
            *flag = test_mm512_insertf32x8();
            break;
        case UT_MM512_INSERTF64X4:
            ret = "MM512_INSERTF64X4";
            *flag = test_mm512_insertf64x4();
            break;
        case UT_MM512_INSERTI32X8:
            ret = "MM512_INSERTI32X8";
            *flag = test_mm512_inserti32x8();
            break;
        case UT_MM512_INSERTI64X4:
            ret = "MM512_INSERTI64X4";
            *flag = test_mm512_inserti64x4();
            break;
        case UT_MM512_PERMUTEXVAR_EPI32:
            ret = "MM512_PERMUTEXVAR_EPI32";
            *flag = test_mm512_permutexvar_epi32();
            break;
        case UT_MM512_PERMUTEX2VAR_EPI32:
            ret = "MM512_PERMUTEX2VAR_EPI32";
            *flag = test_mm512_permutex2var_epi32();
            break;
        case UT_MM256_CMP_PD:
            ret = "MM256_CMP_PD";
            *flag = test_mm256_cmp_pd();
            break;
        case UT_MM256_CMP_PS:
            ret = "MM256_CMP_PS";
            *flag = test_mm256_cmp_ps();
            break;
        case UT_MM512_CMP_PD_MASK:
            ret = "MM512_CMP_PD_MASK";
            *flag = test_mm512_cmp_pd_mask();
            break;
        case UT_MM512_CMP_PS_MASK:
            ret = "MM512_CMP_PS_MASK";
            *flag = test_mm512_cmp_ps_mask();
            break;
        case UT_MM_LOAD_EPI32:
            ret = "MM_LOAD_EPI32";
            *flag = test_mm_load_epi32();
            break;
        case UT_MM_LOAD_EPI64:
            ret = "MM_LOAD_EPI64";
            *flag = test_mm_load_epi64();
            break;
        case UT_MM_LOAD_SI128:
            ret = "MM_LOAD_SI128";
            *flag = test_mm_load_si128();
            break;
        case UT_MM_LOAD_PD:
            ret = "MM_LOAD_PD";
            *flag = test_mm_load_pd();
            break;
        case UT_MM_LOAD_PS:
            ret = "MM_LOAD_PS";
            *flag = test_mm_load_ps();
            break;
        case UT_MM256_LOAD_EPI32:
            ret = "MM256_LOAD_EPI32";
            *flag = test_mm256_load_epi32();
            break;
        case UT_MM256_LOAD_EPI64:
            ret = "MM256_LOAD_EPI64";
            *flag = test_mm256_load_epi64();
            break;
        case UT_MM256_LOAD_PD:
            ret = "MM256_LOAD_PD";
            *flag = test_mm256_load_pd();
            break;
        case UT_MM256_LOAD_PS:
            ret = "MM256_LOAD_PS";
            *flag = test_mm256_load_ps();
            break;
        case UT_MM512_LOAD_EPI32:
            ret = "MM512_LOAD_EPI32";
            *flag = test_mm512_load_epi32();
            break;
        case UT_MM512_LOAD_EPI64:
            ret = "MM512_LOAD_EPI64";
            *flag = test_mm512_load_epi64();
            break;
        case UT_MM512_LOAD_PD:
            ret = "MM512_LOAD_PD";
            *flag = test_mm512_load_pd();
            break;
        case UT_MM512_LOAD_PS:
            ret = "MM512_LOAD_PS";
            *flag = test_mm512_load_ps();
            break;
        case UT_MM_STORE_EPI32:
            ret = "MM_STORE_EPI32";
            *flag = test_mm_store_epi32();
            break;
        case UT_MM_STORE_EPI64:
            ret = "MM_STORE_EPI64";
            *flag = test_mm_store_epi64();
            break;
        case UT_MM_STORE_SI128:
            ret = "MM_STORE_SI128";
            *flag = test_mm_store_si128();
            break;
        case UT_MM_STORE_PD:
            ret = "MM_STORE_PD";
            *flag = test_mm_store_pd();
            break;
        case UT_MM_STORE_PS:
            ret = "MM_STORE_PS";
            *flag = test_mm_store_ps();
            break;
        case UT_MM256_STORE_EPI32:
            ret = "MM256_STORE_EPI32";
            *flag = test_mm256_store_epi32();
            break;
        case UT_MM256_STORE_EPI64:
            ret = "MM256_STORE_EPI64";
            *flag = test_mm256_store_epi64();
            break;
        case UT_MM256_STORE_PD:
            ret = "MM256_STORE_PD";
            *flag = test_mm256_store_pd();
            break;
        case UT_MM256_STORE_PS:
            ret = "MM256_STORE_PS";
            *flag = test_mm256_store_ps();
            break;
        case UT_MM512_STORE_EPI32:
            ret = "MM512_STORE_EPI32";
            *flag = test_mm512_store_epi32();
            break;
        case UT_MM512_STORE_EPI64:
            ret = "MM512_STORE_EPI64";
            *flag = test_mm512_store_epi64();
            break;
        case UT_MM512_STORE_PD:
            ret = "MM512_STORE_PD";
            *flag = test_mm512_store_pd();
            break;
        case UT_MM512_STORE_PS:
            ret = "MM512_STORE_PS";
            *flag = test_mm512_store_ps();
            break;
		case UT_MM_MAX_EPU8:
            ret = "MM_MAX_EPU8";
            *flag = test_mm_max_epu8();
            break;
        case UT_MM_MIN_EPU8:
            ret = "MM_MIN_EPU8";
            *flag = test_mm_min_epu8();
            break;
        case UT_MM256_MAX_EPI32:
            ret = "MM256_MAX_EPI32";
            *flag = test_mm256_max_epi32();
            break;
        case UT_MM512_MAX_EPI32:
            ret = "MM512_MAX_EPI32";
            *flag = test_mm512_max_epi32();
            break;
        case UT_MM_PACKS_EPI16:
            ret = "MM_PACKS_EPI16";
            *flag = test_mm_packs_epi16();
            break;
        case UT_MM_PACKS_EPI32:
            ret = "MM_PACKS_EPI32";
            *flag = test_mm_packs_epi32();
            break;
        case UT_MM256_PACKS_EPI32:
            ret = "MM256_PACKS_EPI32";
            *flag = test_mm256_packs_epi32();
            break;
        case UT_MM512_PACKS_EPI32:
            ret = "MM512_PACKS_EPI32";
            *flag = test_mm512_packs_epi32();
            break;
        case UT_MM_MALLOC:
            ret = "MM_MALLOC";
            *flag = test_mm_malloc();
            break;
        default:
            break;
    }
    return ret;
}

int IsEqualFloat32x4(__m128 a, const float32_t *x, float epsilon)
{
    float e0 = fabs(vgetq_lane_f32(a, 0) - x[0]);
    float e1 = fabs(vgetq_lane_f32(a, 1) - x[1]);
    float e2 = fabs(vgetq_lane_f32(a, 2) - x[2]);
    float e3 = fabs(vgetq_lane_f32(a, 3) - x[3]);
    ASSERT_RETURN(e0 < epsilon);
    ASSERT_RETURN(e1 < epsilon);
    ASSERT_RETURN(e2 < epsilon);
    ASSERT_RETURN(e3 < epsilon);
    return TRUE;
}
int IsEqualFloat64x2(__m128d a, const float64_t *x, float epsilon)
{
    double e0 = fabs(vgetq_lane_f64(a, 0) - x[0]);
    double e1 = fabs(vgetq_lane_f64(a, 1) - x[1]);
    ASSERT_RETURN(e0 < epsilon);
    ASSERT_RETURN(e1 < epsilon);
    return TRUE;
}

int IsEqualFloat32x8(__m256 a, const float *x, float eps)
{
    __m128 tmp;
    for (unsigned int i = 0; i < sizeof(__m256) / sizeof(__m128); i++) {
        tmp = a.vect_f32[i];
        ASSERT_RETURN(IsEqualFloat32x4(tmp, x + i * sizeof(__m128) / sizeof(float), eps));
    }
    return TRUE;
}

int IsEqualFloat64x4(__m256d a, const double *x, float eps)
{
    __m128d tmp;
    for (unsigned int i = 0; i < sizeof(__m256d) / sizeof(__m128d); i++) {
        tmp = a.vect_f64[i];
        ASSERT_RETURN(IsEqualFloat64x2(tmp, x + i * sizeof(__m128d) / sizeof(double), eps));
    }
    return TRUE;
}

int IsEqualFloat32x16(__m512 a, const float *x, float eps)
{
    __m128 tmp;
    for (unsigned int i = 0; i < sizeof(__m512) / sizeof(__m128); i++) {
        tmp = a.vect_f32[i];
        ASSERT_RETURN(IsEqualFloat32x4(tmp, x + i * sizeof(__m128) / sizeof(float), eps));
    }
    return TRUE;
}

int IsEqualFloat64x8(__m512d a, const double *x, float eps)
{
    __m128d tmp;
    for (unsigned int i = 0; i < sizeof(__m512d) / sizeof(__m128d); i++) {
        tmp = a.vect_f64[i];
        ASSERT_RETURN(IsEqualFloat64x2(tmp, x + i * sizeof(__m128d) / sizeof(double), eps));
    }
    return TRUE;
}

int test_mm_popcnt_u32()
{
    unsigned int a = 1587;
    int expect = 6;
    int res = _mm_popcnt_u32(a);
    return (expect == res);
}
int test_mm_popcnt_u64()
{
    unsigned __int64 a = 34359738516;
    __int64 expect = 4;
    __int64 res = _mm_popcnt_u64(a);
    return (expect == res);
}
int test_mm256_div_epi8()
{
    int8_t *a = g_test_mm256_div_epi8_data.a;
    int8_t *b = g_test_mm256_div_epi8_data.b;
    int8_t *expect = g_test_mm256_div_epi8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m256i res = _mm256_div_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm256_div_epi16()
{
    int16_t *a = g_test_mm256_div_epi16_data.a;
    int16_t *b = g_test_mm256_div_epi16_data.b;
    int16_t *expect = g_test_mm256_div_epi16_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m256i res = _mm256_div_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm256_div_epi32()
{
    int32_t *a = g_test_mm256_div_epi32_data.a;
    int32_t *b = g_test_mm256_div_epi32_data.b;
    int32_t *expect = g_test_mm256_div_epi32_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_div_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm256_div_epi64()
{
    int64_t *a = g_test_mm256_div_epi64_data.a;
    int64_t *b = g_test_mm256_div_epi64_data.b;
    int64_t *expect = g_test_mm256_div_epi64_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    __m256i res = _mm256_div_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm256_div_epu8()
{
    uint8_t *a = g_test_mm256_div_epu8_data.a;
    uint8_t *b = g_test_mm256_div_epu8_data.b;
    uint8_t *expect = g_test_mm256_div_epu8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_u8[iCount] = vld1q_u8(a + iCount * 16);
        mb.vect_u8[iCount] = vld1q_u8(b + iCount * 16);
    }
    __m256i res = _mm256_div_epu8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm256_div_epu16()
{
    uint16_t *a = g_test_mm256_div_epu16_data.a;
    uint16_t *b = g_test_mm256_div_epu16_data.b;
    uint16_t *expect = g_test_mm256_div_epu16_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_u16[iCount] = vld1q_u16(a + iCount * 8);
        mb.vect_u16[iCount] = vld1q_u16(b + iCount * 8);
    }
    __m256i res = _mm256_div_epu16(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm256_div_epu32()
{
    uint32_t *a = g_test_mm256_div_epu32_data.a;
    uint32_t *b = g_test_mm256_div_epu32_data.b;
    uint32_t *expect = g_test_mm256_div_epu32_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_u32[iCount] = vld1q_u32(a + iCount * 4);
        mb.vect_u32[iCount] = vld1q_u32(b + iCount * 4);
    }
    __m256i res = _mm256_div_epu32(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm256_div_epu64()
{
    uint64_t *a = g_test_mm256_div_epu64_data.a;
    uint64_t *b = g_test_mm256_div_epu64_data.b;
    uint64_t *expect = g_test_mm256_div_epu64_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_u64[iCount] = vld1q_u64(a + iCount * 2);
        mb.vect_u64[iCount] = vld1q_u64(b + iCount * 2);
    }
    __m256i res = _mm256_div_epu64(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm256_div_pd()
{
    float64_t *a = g_test_mm256_div_pd_data.a;
    float64_t *b = g_test_mm256_div_pd_data.b;
    float64_t *expect = g_test_mm256_div_pd_data.expect;
    int iCount;
    __m256d ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m256d res = _mm256_div_pd(ma, mb);

    return IsEqualFloat64x4(res, expect, DEFAULT_EPSILON_F64);
}
int test_mm256_div_ps()
{
    float32_t *a = g_test_mm256_div_ps_data.a;
    float32_t *b = g_test_mm256_div_ps_data.b;
    float32_t *expect = g_test_mm256_div_ps_data.expect;
    int iCount;
    __m256 ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m256 res = _mm256_div_ps(ma, mb);

    return IsEqualFloat32x8(res, expect, DEFAULT_EPSILON_F32);
}
int test_mm512_div_epi8()
{
    int8_t *a = g_test_mm512_div_epi8_data.a;
    int8_t *b = g_test_mm512_div_epi8_data.b;
    int8_t *expect = g_test_mm512_div_epi8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m512i res = _mm512_div_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm512_div_epi16()
{
    int16_t *a = g_test_mm512_div_epi16_data.a;
    int16_t *b = g_test_mm512_div_epi16_data.b;
    int16_t *expect = g_test_mm512_div_epi16_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m512i res = _mm512_div_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm512_div_epi32()
{
    int32_t *a = g_test_mm512_div_epi32_data.a;
    int32_t *b = g_test_mm512_div_epi32_data.b;
    int32_t *expect = g_test_mm512_div_epi32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_div_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm512_div_epi64()
{
    int64_t *a = g_test_mm512_div_epi64_data.a;
    int64_t *b = g_test_mm512_div_epi64_data.b;
    int64_t *expect = g_test_mm512_div_epi64_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    __m512i res = _mm512_div_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm512_div_epu8()
{
    uint8_t *a = g_test_mm512_div_epu8_data.a;
    uint8_t *b = g_test_mm512_div_epu8_data.b;
    uint8_t *expect = g_test_mm512_div_epu8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_u8[iCount] = vld1q_u8(a + iCount * 16);
        mb.vect_u8[iCount] = vld1q_u8(b + iCount * 16);
    }
    __m512i res = _mm512_div_epu8(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm512_div_epu16()
{
    uint16_t *a = g_test_mm512_div_epu16_data.a;
    uint16_t *b = g_test_mm512_div_epu16_data.b;
    uint16_t *expect = g_test_mm512_div_epu16_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_u16[iCount] = vld1q_u16(a + iCount * 8);
        mb.vect_u16[iCount] = vld1q_u16(b + iCount * 8);
    }
    __m512i res = _mm512_div_epu16(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm512_div_epu32()
{
    uint32_t *a = g_test_mm512_div_epu32_data.a;
    uint32_t *b = g_test_mm512_div_epu32_data.b;
    uint32_t *expect = g_test_mm512_div_epu32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_u32[iCount] = vld1q_u32(a + iCount * 4);
        mb.vect_u32[iCount] = vld1q_u32(b + iCount * 4);
    }
    __m512i res = _mm512_div_epu32(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm512_div_epu64()
{
    uint64_t *a = g_test_mm512_div_epu64_data.a;
    uint64_t *b = g_test_mm512_div_epu64_data.b;
    uint64_t *expect = g_test_mm512_div_epu64_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_u64[iCount] = vld1q_u64(a + iCount * 2);
        mb.vect_u64[iCount] = vld1q_u64(b + iCount * 2);
    }
    __m512i res = _mm512_div_epu64(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm512_div_pd()
{
    float64_t *a = g_test_mm512_div_pd_data.a;
    float64_t *b = g_test_mm512_div_pd_data.b;
    float64_t *expect = g_test_mm512_div_pd_data.expect;
    int iCount;
    __m512d ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m512d res = _mm512_div_pd(ma, mb);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}
int test_mm512_div_ps()
{
    float32_t *a = g_test_mm512_div_ps_data.a;
    float32_t *b = g_test_mm512_div_ps_data.b;
    float32_t *expect = g_test_mm512_div_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m512 res = _mm512_div_ps(ma, mb);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_div_round_ps()
{
    float32_t *a = g_test_mm512_div_round_ps_data.a;
    float32_t *b = g_test_mm512_div_round_ps_data.b;
    int rounding = g_test_mm512_div_round_ps_data.rounding;
    float32_t *expect = g_test_mm512_div_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m512 res = _mm512_div_round_ps(ma, mb, rounding);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}
int test_mm512_div_round_pd()
{
    float64_t *a = g_test_mm512_div_pd_data.a;
    float64_t *b = g_test_mm512_div_pd_data.b;
    int rounding = g_test_mm512_div_round_pd_data.rounding;
    float64_t *expect = g_test_mm512_div_pd_data.expect;
    int iCount;
    __m512d ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m512d res = _mm512_div_round_pd(ma, mb, rounding);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}
int test_mm256_add_epi8()
{
    int8_t *a = g_test_mm256_add_epi8_data.a;
    int8_t *b = g_test_mm256_add_epi8_data.b;
    int8_t *expect = g_test_mm256_add_epi8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m256i res = _mm256_add_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_add_epi16()
{
    int16_t *a = g_test_mm256_add_epi16_data.a;
    int16_t *b = g_test_mm256_add_epi16_data.b;
    int16_t *expect = g_test_mm256_add_epi16_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m256i res = _mm256_add_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_add_epi32()
{
    int32_t *a = g_test_mm256_add_epi32_data.a;
    int32_t *b = g_test_mm256_add_epi32_data.b;
    int32_t *expect = g_test_mm256_add_epi32_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_add_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_add_epi64()
{
    int64_t *a = g_test_mm256_add_epi64_data.a;
    int64_t *b = g_test_mm256_add_epi64_data.b;
    int64_t *expect = g_test_mm256_add_epi64_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    __m256i res = _mm256_add_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_add_epi8()
{
    int8_t *a = g_test_mm512_add_epi8_data.a;
    int8_t *b = g_test_mm512_add_epi8_data.b;
    int8_t *expect = g_test_mm512_add_epi8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m512i res = _mm512_add_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_add_epi16()
{
    int16_t *a = g_test_mm512_add_epi16_data.a;
    int16_t *b = g_test_mm512_add_epi16_data.b;
    int16_t *expect = g_test_mm512_add_epi16_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m512i res = _mm512_add_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_add_epi32()
{
    int32_t *a = g_test_mm512_add_epi32_data.a;
    int32_t *b = g_test_mm512_add_epi32_data.b;
    int32_t *expect = g_test_mm512_add_epi32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_add_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_add_epi64()
{
    int64_t *a = g_test_mm512_add_epi64_data.a;
    int64_t *b = g_test_mm512_add_epi64_data.b;
    int64_t *expect = g_test_mm512_add_epi64_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    __m512i res = _mm512_add_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm_adds_epu8()
{
    uint8_t *a = g_test_mm_adds_epu8_data.a;
    uint8_t *b = g_test_mm_adds_epu8_data.b;
    uint8_t *expect = g_test_mm_adds_epu8_data.expect;
    __m128i ma, mb;
    ma.vect_u8 = vld1q_u8(a);
    mb.vect_u8 = vld1q_u8(b);
    __m128i res = _mm_adds_epu8(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm256_adds_epi8()
{
    int8_t *a = g_test_mm256_adds_epi8_data.a;
    int8_t *b = g_test_mm256_adds_epi8_data.b;
    int8_t *expect = g_test_mm256_adds_epi8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m256i res = _mm256_adds_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_adds_epi16()
{
    int16_t *a = g_test_mm256_adds_epi16_data.a;
    int16_t *b = g_test_mm256_adds_epi16_data.b;
    int16_t *expect = g_test_mm256_adds_epi16_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m256i res = _mm256_adds_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_adds_epu8()
{
    uint8_t *a = g_test_mm256_adds_epu8_data.a;
    uint8_t *b = g_test_mm256_adds_epu8_data.b;
    uint8_t *expect = g_test_mm256_adds_epu8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_u8[iCount] = vld1q_u8(a + iCount * 16);
        mb.vect_u8[iCount] = vld1q_u8(b + iCount * 16);
    }
    __m256i res = _mm256_adds_epu8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_adds_epu16()
{
    uint16_t *a = g_test_mm256_adds_epu16_data.a;
    uint16_t *b = g_test_mm256_adds_epu16_data.b;
    uint16_t *expect = g_test_mm256_adds_epu16_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_u16[iCount] = vld1q_u16(a + iCount * 8);
        mb.vect_u16[iCount] = vld1q_u16(b + iCount * 8);
    }
    __m256i res = _mm256_adds_epu16(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_adds_epi8()
{
    int8_t *a = g_test_mm512_adds_epi8_data.a;
    int8_t *b = g_test_mm512_adds_epi8_data.b;
    int8_t *expect = g_test_mm512_adds_epi8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m512i res = _mm512_adds_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_adds_epi16()
{
    int16_t *a = g_test_mm512_adds_epi16_data.a;
    int16_t *b = g_test_mm512_adds_epi16_data.b;
    int16_t *expect = g_test_mm512_adds_epi16_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m512i res = _mm512_adds_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_adds_epu8()
{
    uint8_t *a = g_test_mm512_adds_epu8_data.a;
    uint8_t *b = g_test_mm512_adds_epu8_data.b;
    uint8_t *expect = g_test_mm512_adds_epu8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_u8[iCount] = vld1q_u8(a + iCount * 16);
        mb.vect_u8[iCount] = vld1q_u8(b + iCount * 16);
    }
    __m512i res = _mm512_adds_epu8(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_adds_epu16()
{
    uint16_t *a = g_test_mm512_adds_epu16_data.a;
    uint16_t *b = g_test_mm512_adds_epu16_data.b;
    uint16_t *expect = g_test_mm512_adds_epu16_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_u16[iCount] = vld1q_u16(a + iCount * 8);
        mb.vect_u16[iCount] = vld1q_u16(b + iCount * 8);
    }
    __m512i res = _mm512_adds_epu16(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm256_add_ps()
{
    float32_t *a = g_test_mm256_add_ps_data.a;
    float32_t *b = g_test_mm256_add_ps_data.b;
    float32_t *expect = g_test_mm256_add_ps_data.expect;
    int iCount;
    __m256 ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m256 res = _mm256_add_ps(ma, mb);

    return IsEqualFloat32x8(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm256_add_pd()
{
    float64_t *a = g_test_mm256_add_pd_data.a;
    float64_t *b = g_test_mm256_add_pd_data.b;
    float64_t *expect = g_test_mm256_add_pd_data.expect;
    int iCount;
    __m256d ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m256d res = _mm256_add_pd(ma, mb);

    return IsEqualFloat64x4(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_add_ps()
{
    float32_t *a = g_test_mm512_add_ps_data.a;
    float32_t *b = g_test_mm512_add_ps_data.b;
    float32_t *expect = g_test_mm512_add_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m512 res = _mm512_add_ps(ma, mb);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_add_pd()
{
    float64_t *a = g_test_mm512_add_pd_data.a;
    float64_t *b = g_test_mm512_add_pd_data.b;
    float64_t *expect = g_test_mm512_add_pd_data.expect;
    int iCount;
    __m512d ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m512d res = _mm512_add_pd(ma, mb);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_add_round_ps()
{
    float32_t *a = g_test_mm512_add_round_ps_data.a;
    float32_t *b = g_test_mm512_add_round_ps_data.b;
    int rounding = g_test_mm512_add_round_ps_data.rounding;
    float32_t *expect = g_test_mm512_add_round_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m512 res = _mm512_add_round_ps(ma, mb, rounding);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_add_round_pd()
{
    float64_t *a = g_test_mm512_add_round_pd_data.a;
    float64_t *b = g_test_mm512_add_round_pd_data.b;
    int rounding = g_test_mm512_add_round_pd_data.rounding;
    float64_t *expect = g_test_mm512_add_round_pd_data.expect;
    int iCount;
    __m512d ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m512d res = _mm512_add_round_pd(ma, mb, rounding);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_addn_ps()
{
    float32_t *a = g_test_mm512_addn_ps_data.a;
    float32_t *b = g_test_mm512_addn_ps_data.b;
    float32_t *expect = g_test_mm512_addn_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m512 res = _mm512_addn_ps(ma, mb);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_addn_pd()
{
    float64_t *a = g_test_mm512_addn_pd_data.a;
    float64_t *b = g_test_mm512_addn_pd_data.b;
    float64_t *expect = g_test_mm512_addn_pd_data.expect;
    int iCount;
    __m512d ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m512d res = _mm512_addn_pd(ma, mb);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_addn_round_ps()
{
    float32_t *a = g_test_mm512_addn_round_ps_data.a;
    float32_t *b = g_test_mm512_addn_round_ps_data.b;
    int rounding = g_test_mm512_addn_round_ps_data.rounding;
    float32_t *expect = g_test_mm512_addn_round_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m512 res = _mm512_addn_round_ps(ma, mb, rounding);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_addn_round_pd()
{
    float64_t *a = g_test_mm512_addn_round_pd_data.a;
    float64_t *b = g_test_mm512_addn_round_pd_data.b;
    int rounding = g_test_mm512_addn_round_pd_data.rounding;
    float64_t *expect = g_test_mm512_addn_round_pd_data.expect;
    int iCount;
    __m512d ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m512d res = _mm512_addn_round_pd(ma, mb, rounding);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_addsetc_epi32()
{
    int32_t *a = g_test_mm512_addsetc_epi32_data.a;
    int32_t *b = g_test_mm512_addsetc_epi32_data.b;
    int32_t *expect = g_test_mm512_addsetc_epi32_data.expect;
    __mmask16 expect_sign = g_test_mm512_addsetc_epi32_data.sign;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __mmask16 sign;
    __m512i res = _mm512_addsetc_epi32(ma, mb, &sign);
    return comp_return(expect, &res, sizeof(__m512i)) && (expect_sign == sign);
}

int test_mm512_addsets_epi32()
{
    int32_t *a = g_test_mm512_addsets_epi32_data.a;
    int32_t *b = g_test_mm512_addsets_epi32_data.b;
    int32_t *expect = g_test_mm512_addsets_epi32_data.expect;
    __mmask16 expect_sign = g_test_mm512_addsets_epi32_data.sign;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __mmask16 sign;
    __m512i res = _mm512_addsets_epi32(ma, mb, &sign);
    return comp_return(expect, &res, sizeof(__m512i)) && (expect_sign == sign);
}

int test_mm512_addsets_ps()
{
    float32_t *a = g_test_mm512_addsets_ps_data.a;
    float32_t *b = g_test_mm512_addsets_ps_data.b;
    float32_t *expect = g_test_mm512_addsets_ps_data.expect;
    __mmask16 expect_sign = g_test_mm512_addsets_ps_data.sign;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __mmask16 sign;
    __m512 res = _mm512_addsets_ps(ma, mb, &sign);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32) && (expect_sign == sign);
}

int test_mm512_addsets_round_ps()
{
    float32_t *a = g_test_mm512_addsets_round_ps_data.a;
    float32_t *b = g_test_mm512_addsets_round_ps_data.b;
    int rounding = g_test_mm512_addsets_round_ps_data.rounding;
    float32_t *expect = g_test_mm512_addsets_round_ps_data.expect;
    __mmask16 expect_sign = g_test_mm512_addsets_round_ps_data.sign;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __mmask16 sign;
    __m512 res = _mm512_addsets_round_ps(ma, mb, &sign, rounding);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32) && (expect_sign == sign);
}

int test_mm256_addsub_ps()
{
    float32_t *a = g_test_mm256_addsub_ps_data.a;
    float32_t *b = g_test_mm256_addsub_ps_data.b;
    float32_t *expect = g_test_mm256_addsub_ps_data.expect;
    int iCount;
    __m256 ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m256 res = _mm256_addsub_ps(ma, mb);

    return IsEqualFloat32x8(res, expect, DEFAULT_EPSILON_F32);
}
int test_mm256_addsub_pd()
{
    float64_t *a = g_test_mm256_addsub_pd_data.a;
    float64_t *b = g_test_mm256_addsub_pd_data.b;
    float64_t *expect = g_test_mm256_addsub_pd_data.expect;
    int iCount;
    __m256d ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m256d res = _mm256_addsub_pd(ma, mb);

    return IsEqualFloat64x4(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm_sub_epi8()
{
    int8_t *a = g_test_mm_sub_epi8_data.a;
    int8_t *b = g_test_mm_sub_epi8_data.b;
    int8_t *expect = g_test_mm_sub_epi8_data.expect;
    __m128i ma, mb;
    ma.vect_s8 = vld1q_s8(a);
    mb.vect_s8 = vld1q_s8(b);
    __m128i res = _mm_sub_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm256_sub_epi16()
{
    int16_t *a = g_test_mm256_sub_epi16_data.a;
    int16_t *b = g_test_mm256_sub_epi16_data.b;
    int16_t *expect = g_test_mm256_sub_epi16_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m256i res = _mm256_sub_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_sub_epi32()
{
    int32_t *a = g_test_mm256_sub_epi32_data.a;
    int32_t *b = g_test_mm256_sub_epi32_data.b;
    int32_t *expect = g_test_mm256_sub_epi32_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_sub_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_sub_epi64()
{
    int64_t *a = g_test_mm256_sub_epi64_data.a;
    int64_t *b = g_test_mm256_sub_epi64_data.b;
    int64_t *expect = g_test_mm256_sub_epi64_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    __m256i res = _mm256_sub_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_sub_epi8()
{
    int8_t *a = g_test_mm256_sub_epi8_data.a;
    int8_t *b = g_test_mm256_sub_epi8_data.b;
    int8_t *expect = g_test_mm256_sub_epi8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m256i res = _mm256_sub_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_sub_pd()
{
    float64_t *a = g_test_mm256_sub_pd_data.a;
    float64_t *b = g_test_mm256_sub_pd_data.b;
    float64_t *expect = g_test_mm256_sub_pd_data.expect;
    int iCount;
    __m256d ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m256d res = _mm256_sub_pd(ma, mb);

    return IsEqualFloat64x4(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm256_sub_ps()
{
    float32_t *a = g_test_mm256_sub_ps_data.a;
    float32_t *b = g_test_mm256_sub_ps_data.b;
    float32_t *expect = g_test_mm256_sub_ps_data.expect;
    int iCount;
    __m256 ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m256 res = _mm256_sub_ps(ma, mb);

    return IsEqualFloat32x8(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_sub_epi16()
{
    int16_t *a = g_test_mm512_sub_epi16_data.a;
    int16_t *b = g_test_mm512_sub_epi16_data.b;
    int16_t *expect = g_test_mm512_sub_epi16_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m512i res = _mm512_sub_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_sub_epi32()
{
    int32_t *a = g_test_mm512_sub_epi32_data.a;
    int32_t *b = g_test_mm512_sub_epi32_data.b;
    int32_t *expect = g_test_mm512_sub_epi32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_sub_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_sub_epi64()
{
    int64_t *a = g_test_mm512_sub_epi64_data.a;
    int64_t *b = g_test_mm512_sub_epi64_data.b;
    int64_t *expect = g_test_mm512_sub_epi64_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    __m512i res = _mm512_sub_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_sub_epi8()
{
    int8_t *a = g_test_mm512_sub_epi8_data.a;
    int8_t *b = g_test_mm512_sub_epi8_data.b;
    int8_t *expect = g_test_mm512_sub_epi8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m512i res = _mm512_sub_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_sub_pd()
{
    float64_t *a = g_test_mm512_sub_pd_data.a;
    float64_t *b = g_test_mm512_sub_pd_data.b;
    float64_t *expect = g_test_mm512_sub_pd_data.expect;
    int iCount;
    __m512d ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m512d res = _mm512_sub_pd(ma, mb);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_bslli_epi128()
{
    int64_t *a = g_test_mm512_bslli_epi128_data.a;
    int64_t *expect = g_test_mm512_bslli_epi128_data.expect;
    int iCount;
    __m512i ma;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    __m512i res = _mm512_bslli_epi128(ma, 8);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_bsrli_epi128()
{
    int64_t *a = g_test_mm512_bsrli_epi128_data.a;
    int64_t *expect = g_test_mm512_bsrli_epi128_data.expect;
    int iCount;
    __m512i ma;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    __m512i res = _mm512_bsrli_epi128(ma, 4);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm_sll_epi64()
{
    int64_t a[2] = {1, -1};
    int64_t b[2] = {2, 2};
    int64_t expect[2] = {4, -4};
    __m128i ma, mb;
    ma.vect_s64 = vld1q_s64(a);
    mb.vect_s64 = vld1q_s64(b);
    __m128i res = _mm_sll_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_slli_si128()
{
    int64_t *a = g_test_mm_slli_si128_data.a;
    uint64_t *expect = g_test_mm_slli_si128_data.expect;
    __m128i ma;
    ma.vect_s64 = vld1q_s64(a);
    __m128i res = _mm_slli_si128(ma, 2);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_srli_si128()
{
    int64_t *a = g_test_mm_srli_si128_data.a;
    uint64_t *expect = g_test_mm_srli_si128_data.expect;
    __m128i ma;
    ma.vect_s64 = vld1q_s64(a);
    __m128i res = _mm_srli_si128(ma, 2);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_slli_epi32()
{
    int32_t *a = g_test_mm_slli_epi32_data.a;
    int b = g_test_mm_slli_epi32_data.b;
    int32_t *expect = g_test_mm_slli_epi32_data.expect;
    __m128i ma;
    ma.vect_s32 = vld1q_s32(a);
    __m128i res = _mm_slli_epi32(ma, b);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_slli_epi64()
{
    int64_t *a = g_test_mm_slli_epi64_data.a;
    int b = g_test_mm_slli_epi64_data.b;
    int64_t *expect = g_test_mm_slli_epi64_data.expect;
    __m128i ma;
    ma.vect_s64 = vld1q_s64(a);
    __m128i res = _mm_slli_epi64(ma, b);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_srli_epi64()
{
    int64_t *a = g_test_mm_srli_epi64_data.a;
    int b = g_test_mm_srli_epi64_data.b;
    uint64_t *expect = g_test_mm_srli_epi64_data.expect;
    __m128i ma;
    ma.vect_s64 = vld1q_s64(a);
    __m128i res = _mm_srli_epi64(ma, b);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm256_sll_epi32()
{
    int32_t *a = g_test_mm256_sll_epi32_data.a;
    int64_t *b = g_test_mm256_sll_epi32_data.b;
    int32_t *expect = g_test_mm256_sll_epi32_data.expect;
    int iCount;
    __m256i ma;
    __m128i mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    mb.vect_s64 = vld1q_s64(b);
    __m256i res = _mm256_sll_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm256_sll_epi64()
{
    int64_t *a = g_test_mm256_sll_epi64_data.a;
    int64_t *b = g_test_mm256_sll_epi64_data.b;
    int64_t *expect = g_test_mm256_sll_epi64_data.expect;
    int iCount;
    __m256i ma;
    __m128i mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    mb.vect_s64 = vld1q_s64(b);
    __m256i res = _mm256_sll_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm512_sll_epi64()
{
    int64_t *a = g_test_mm512_sll_epi64_data.a;
    int64_t *b = g_test_mm512_sll_epi64_data.b;
    int64_t *expect = g_test_mm512_sll_epi64_data.expect;
    int iCount;
    __m512i ma;
    __m128i mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    mb.vect_s64 = vld1q_s64(b);
    __m512i res = _mm512_sll_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm256_slli_epi32()
{
    int32_t *a = g_test_mm256_slli_epi32_data.a;
    int b = g_test_mm256_slli_epi32_data.b;
    int32_t *expect = g_test_mm256_slli_epi32_data.expect;
    int iCount;
    __m256i ma;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    __m256i res = _mm256_slli_epi32(ma, b);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_slli_epi64()
{
    int64_t *a = g_test_mm256_slli_epi64_data.a;
    int b = g_test_mm256_slli_epi64_data.b;
    int64_t *expect = g_test_mm256_slli_epi64_data.expect;
    int iCount;
    __m256i ma;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    __m256i res = _mm256_slli_epi64(ma, b);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm512_slli_epi64()
{
    int64_t *a = g_test_mm512_slli_epi64_data.a;
    unsigned int b = g_test_mm512_slli_epi64_data.b;
    int64_t *expect = g_test_mm512_slli_epi64_data.expect;
    int iCount;
    __m512i ma;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    __m512i res = _mm512_slli_epi64(ma, b);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm256_srli_epi64()
{
    int64_t *a = g_test_mm256_srli_epi64_data.a;
    int b = g_test_mm256_srli_epi64_data.b;
    uint64_t *expect = g_test_mm256_srli_epi64_data.expect;
    int iCount;
    __m256i ma;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    __m256i res = _mm256_srli_epi64(ma, b);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm512_srli_epi64()
{
    int64_t *a = g_test_mm512_srli_epi64_data.a;
    unsigned int b = g_test_mm512_srli_epi64_data.b;
    uint64_t *expect = g_test_mm512_srli_epi64_data.expect;
    int iCount;
    __m512i ma;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    __m512i res = _mm512_srli_epi64(ma, b);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm256_slli_si256()
{
    int64_t *a = g_test_mm256_slli_si256_data.a;
    uint64_t *expect = g_test_mm256_slli_si256_data.expect;
    int iCount;
    __m256i ma;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    __m256i res = _mm256_slli_si256(ma, 2);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_srli_si256()
{
    int64_t *a = g_test_mm256_srli_si256_data.a;
    uint64_t *expect = g_test_mm256_srli_si256_data.expect;
    int iCount;
    __m256i ma;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    __m256i res = _mm256_srli_si256(ma, 2);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_blendv_ps()
{
    float32_t *arr_a = g_test_mm256_blendv_ps_data.a;
    float32_t *arr_b = g_test_mm256_blendv_ps_data.b;
    float32_t *arr_m = g_test_mm256_blendv_ps_data.m;
    float32_t *expect = g_test_mm256_blendv_ps_data.expect;
    __m256 a, b, mask, res;
    int iCount;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        a.vect_f32[iCount] = vld1q_f32(arr_a + iCount * 4);
        b.vect_f32[iCount] = vld1q_f32(arr_b + iCount * 4);
        mask.vect_f32[iCount] = vld1q_f32(arr_m + iCount * 4);
    }

    res = _mm256_blendv_ps(a, b, mask);

    return IsEqualFloat32x8(res, expect, DEFAULT_EPSILON_F32);
}
int test_mm256_blendv_pd()
{
    float64_t *arr_a = g_test_mm256_blendv_pd_data.a;
    float64_t *arr_b = g_test_mm256_blendv_pd_data.b;
    float64_t *arr_m = g_test_mm256_blendv_pd_data.m;
    float64_t *expect = g_test_mm256_blendv_pd_data.expect;
    __m256d a, b, mask, res;
    int iCount;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        a.vect_f64[iCount] = vld1q_f64(arr_a + iCount * 2);
        b.vect_f64[iCount] = vld1q_f64(arr_b + iCount * 2);
        mask.vect_f64[iCount] = vld1q_f64(arr_m + iCount * 2);
    }

    res = _mm256_blendv_pd(a, b, mask);

    return IsEqualFloat64x4(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm256_blend_ps()
{
    float32_t *arr_a = g_test_mm256_blend_ps_data.a;
    float32_t *arr_b = g_test_mm256_blend_ps_data.b;
    int imm = g_test_mm256_blend_ps_data.imm;
    float32_t *expect = g_test_mm256_blend_ps_data.expect;
    __m256 a, b, res;
    int iCount;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        a.vect_f32[iCount] = vld1q_f32(arr_a + iCount * 4);
        b.vect_f32[iCount] = vld1q_f32(arr_b + iCount * 4);
    }

    res = _mm256_blend_ps(a, b, imm);

    return IsEqualFloat32x8(res, expect, DEFAULT_EPSILON_F32);
}
int test_mm256_blend_pd()
{
    float64_t *arr_a = g_test_mm256_blend_pd_data.a;
    float64_t *arr_b = g_test_mm256_blend_pd_data.b;
    int imm = g_test_mm256_blend_pd_data.imm;
    float64_t *expect = g_test_mm256_blend_pd_data.expect;
    __m256d a, b, res;
    int iCount;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        a.vect_f64[iCount] = vld1q_f64(arr_a + iCount * 2);
        b.vect_f64[iCount] = vld1q_f64(arr_b + iCount * 2);
    }

    res = _mm256_blend_pd(a, b, imm);

    return IsEqualFloat64x4(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_mask_blend_epi32()
{
    int32_t *arr_a = g_test_mm512_mask_blend_epi32_data.a;
    int32_t *arr_b = g_test_mm512_mask_blend_epi32_data.b;
    __mmask16 k = g_test_mm512_mask_blend_epi32_data.k;
    int32_t *expect = g_test_mm512_mask_blend_epi32_data.expect;
    __m512i a, b, res;
    int iCount;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        a.vect_s32[iCount] = vld1q_s32(arr_a + iCount * 4);
        b.vect_s32[iCount] = vld1q_s32(arr_b + iCount * 4);
    }

    res = _mm512_mask_blend_epi32(k, a, b);

    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_mask_blend_ps()
{
    float32_t *arr_a = g_test_mm512_mask_blend_ps_data.a;
    float32_t *arr_b = g_test_mm512_mask_blend_ps_data.b;
    __mmask16 k = g_test_mm512_mask_blend_ps_data.k;
    float32_t *expect = g_test_mm512_mask_blend_ps_data.expect;
    __m512 a, b, res;
    int iCount;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        a.vect_f32[iCount] = vld1q_f32(arr_a + iCount * 4);
        b.vect_f32[iCount] = vld1q_f32(arr_b + iCount * 4);
    }

    res = _mm512_mask_blend_ps(k, a, b);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}
int test_mm512_mask_blend_pd()
{
    float64_t *arr_a = g_test_mm512_mask_blend_pd_data.a;
    float64_t *arr_b = g_test_mm512_mask_blend_pd_data.b;
    __mmask8 k = g_test_mm512_mask_blend_pd_data.k;
    float64_t *expect = g_test_mm512_mask_blend_pd_data.expect;
    __m512d a, b, res;
    int iCount;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        a.vect_f64[iCount] = vld1q_f64(arr_a + iCount * 2);
        b.vect_f64[iCount] = vld1q_f64(arr_b + iCount * 2);
    }

    res = _mm512_mask_blend_pd(k, a, b);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_sub_ps()
{
    float32_t *a = g_test_mm512_sub_ps_data.a;
    float32_t *b = g_test_mm512_sub_ps_data.b;
    float32_t *expect = g_test_mm512_sub_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m512 res = _mm512_sub_ps(ma, mb);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm256_subs_epi16()
{
    int16_t *a = g_test_mm256_subs_epi16_data.a;
    int16_t *b = g_test_mm256_subs_epi16_data.b;
    int16_t *expect = g_test_mm256_subs_epi16_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m256i res = _mm256_subs_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_subs_epi8()
{
    int8_t *a = g_test_mm256_subs_epi8_data.a;
    int8_t *b = g_test_mm256_subs_epi8_data.b;
    int8_t *expect = g_test_mm256_subs_epi8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m256i res = _mm256_subs_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm256_subs_epu16()
{
    uint16_t *a = g_test_mm256_subs_epu16_data.a;
    uint16_t *b = g_test_mm256_subs_epu16_data.b;
    uint16_t *expect = g_test_mm256_subs_epu16_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_u16[iCount] = vld1q_u16(a + iCount * 8);
        mb.vect_u16[iCount] = vld1q_u16(b + iCount * 8);
    }
    __m256i res = _mm256_subs_epu16(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_subs_epu8()
{
    uint8_t *a = g_test_mm256_subs_epu8_data.a;
    uint8_t *b = g_test_mm256_subs_epu8_data.b;
    uint8_t *expect = g_test_mm256_subs_epu8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_u8[iCount] = vld1q_u8(a + iCount * 16);
        mb.vect_u8[iCount] = vld1q_u8(b + iCount * 16);
    }
    __m256i res = _mm256_subs_epu8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_subs_epi16()
{
    int16_t *a = g_test_mm512_subs_epi16_data.a;
    int16_t *b = g_test_mm512_subs_epi16_data.b;
    int16_t *expect = g_test_mm512_subs_epi16_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m512i res = _mm512_subs_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_subs_epi8()
{
    int8_t *a = g_test_mm512_subs_epi8_data.a;
    int8_t *b = g_test_mm512_subs_epi8_data.b;
    int8_t *expect = g_test_mm512_subs_epi8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m512i res = _mm512_subs_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_subs_epu16()
{
    uint16_t *a = g_test_mm512_subs_epu16_data.a;
    uint16_t *b = g_test_mm512_subs_epu16_data.b;
    uint16_t *expect = g_test_mm512_subs_epu16_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_u16[iCount] = vld1q_u16(a + iCount * 8);
        mb.vect_u16[iCount] = vld1q_u16(b + iCount * 8);
    }
    __m512i res = _mm512_subs_epu16(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_subs_epu8()
{
    uint8_t *a = g_test_mm512_subs_epu8_data.a;
    uint8_t *b = g_test_mm512_subs_epu8_data.b;
    uint8_t *expect = g_test_mm512_subs_epu8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_u8[iCount] = vld1q_u8(a + iCount * 16);
        mb.vect_u8[iCount] = vld1q_u8(b + iCount * 16);
    }
    __m512i res = _mm512_subs_epu8(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_sub_round_pd()
{
    float64_t *a = g_test_mm512_sub_round_pd_data.a;
    float64_t *b = g_test_mm512_sub_round_pd_data.b;
    float64_t *expect = g_test_mm512_sub_round_pd_data.expect;
    int iCount;
    __m512d ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m512d res = _mm512_sub_round_pd(ma, mb, _MM_FROUND_NO_EXC);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_sub_round_ps()
{
    float32_t *a = g_test_mm512_sub_round_ps_data.a;
    float32_t *b = g_test_mm512_sub_round_ps_data.b;
    float32_t *expect = g_test_mm512_sub_round_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m512 res = _mm512_sub_round_ps(ma, mb, _MM_FROUND_NO_EXC);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_subr_epi32()
{
    int32_t *a = g_test_mm512_subr_epi32_data.a;
    int32_t *b = g_test_mm512_subr_epi32_data.b;
    int32_t *expect = g_test_mm512_subr_epi32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_subr_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_subr_ps()
{
    float32_t *a = g_test_mm512_subr_ps_data.a;
    float32_t *b = g_test_mm512_subr_ps_data.b;
    float32_t *expect = g_test_mm512_subr_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m512 res = _mm512_subr_ps(ma, mb);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_subr_pd()
{
    float64_t *a = g_test_mm512_subr_pd_data.a;
    float64_t *b = g_test_mm512_subr_pd_data.b;
    float64_t *expect = g_test_mm512_subr_pd_data.expect;
    int iCount;
    __m512d ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m512d res = _mm512_subr_pd(ma, mb);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_subr_round_ps()
{
    float32_t *a = g_test_mm512_subr_round_ps_data.a;
    float32_t *b = g_test_mm512_subr_round_ps_data.b;
    float32_t *expect = g_test_mm512_subr_round_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m512 res = _mm512_subr_round_ps(ma, mb, _MM_FROUND_NO_EXC);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}
int test_mm512_subr_round_pd()
{
    float64_t *a = g_test_mm512_subr_round_pd_data.a;
    float64_t *b = g_test_mm512_subr_round_pd_data.b;
    float64_t *expect = g_test_mm512_subr_round_pd_data.expect;
    int iCount;
    __m512d ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m512d res = _mm512_subr_round_pd(ma, mb, _MM_FROUND_NO_EXC);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_subsetb_epi32()
{
    int32_t *a = g_test_mm512_subsetb_epi32_data.a;
    int32_t *b = g_test_mm512_subsetb_epi32_data.b;
    int32_t *expect = g_test_mm512_subsetb_epi32_data.expect;
    __mmask16 expect_borrow = g_test_mm512_subsetb_epi32_data.borrow;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __mmask16 borrow;
    __m512i res = _mm512_subsetb_epi32(ma, mb, &borrow);
    return comp_return(expect, &res, sizeof(__m512i)) && (expect_borrow == borrow);
}
int test_mm512_subrsetb_epi32()
{
    int32_t *a = g_test_mm512_subrsetb_epi32_data.a;
    int32_t *b = g_test_mm512_subrsetb_epi32_data.b;
    int32_t *expect = g_test_mm512_subrsetb_epi32_data.expect;
    __mmask16 expect_borrow = g_test_mm512_subrsetb_epi32_data.borrow;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __mmask16 borrow;
    __m512i res = _mm512_subrsetb_epi32(ma, mb, &borrow);
    return comp_return(expect, &res, sizeof(__m512i)) && (expect_borrow == borrow);
}

int test_mm256_zeroupper()
{
    return 1;
}

int test_mm512_permutexvar_epi64()
{
    int64_t *a = g_test_mm512_permutexvar_epi64_data.a;
    int64_t *b = g_test_mm512_permutexvar_epi64_data.b;
    int64_t *expect = g_test_mm512_permutexvar_epi64_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    } 
    __m512i res = _mm512_permutexvar_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_extracti32x4_epi32()
{
    int32_t *a = g_test_mm512_extracti32x4_epi32_data.a;
    const int imm8 = g_test_mm512_extracti32x4_epi32_data.imm8;
    int32_t *expect = g_test_mm512_extracti32x4_epi32_data.expect;
    int iCount;
    __m512i ma;
    for (iCount = 0; iCount < 4; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    __m128i res = _mm512_extracti32x4_epi32(ma, imm8);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm512_test_epi8_mask()
{
    int8_t *a = g_test_mm512_test_epi8_mask_data.a;
    int8_t *b = g_test_mm512_test_epi8_mask_data.b;
    __mmask64 expect = g_test_mm512_test_epi8_mask_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < 4; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __mmask64 res = _mm512_test_epi8_mask(ma, mb);
    return (res == expect);
}

int test_mm512_test_epi32_mask()
{
    int32_t *a = g_test_mm512_test_epi32_mask_data.a;
    int32_t *b = g_test_mm512_test_epi32_mask_data.b;
    __mmask16 expect = g_test_mm512_test_epi32_mask_data.expect;

    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < 4; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __mmask16 res = _mm512_test_epi32_mask(ma, mb);
    return (res == expect);
}

int test_mm512_test_epi64_mask()
{
    int64_t *a = g_test_mm512_test_epi64_mask_data.a;
    int64_t *b = g_test_mm512_test_epi64_mask_data.b;
    __mmask8 expect = g_test_mm512_test_epi64_mask_data.expect;

    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < 4; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    __mmask8 res = _mm512_test_epi64_mask(ma, mb);
    return (res == expect);
}

int test_mm256_mul_epi32()
{
    int32_t *a = g_test_mm256_mul_epi32_data.a;
    int32_t *b = g_test_mm256_mul_epi32_data.b;
    int32_t *expect = g_test_mm256_mul_epi32_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < 2; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_mul_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_mul_epu32()
{
    uint32_t *a = g_test_mm256_mul_epu32_data.a;
    uint32_t *b = g_test_mm256_mul_epu32_data.b;
    uint32_t *expect = g_test_mm256_mul_epu32_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < 2; iCount++) {
        ma.vect_u32[iCount] = vld1q_u32(a + iCount * 4);
        mb.vect_u32[iCount] = vld1q_u32(b + iCount * 4);
    }
    __m256i res = _mm256_mul_epu32(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_mul_pd()
{
    float64_t *a = g_test_mm256_mul_pd_data.a;
    float64_t *b = g_test_mm256_mul_pd_data.b;
    float64_t *expect = g_test_mm256_mul_pd_data.expect;
    int iCount;
    __m256d ma, mb;
    for (iCount = 0; iCount < 2; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m256d res = _mm256_mul_pd(ma, mb);

    return IsEqualFloat64x4(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm256_mul_ps()
{
    float32_t *a = g_test_mm256_mul_ps_data.a;
    float32_t *b = g_test_mm256_mul_ps_data.b;
    float32_t *expect = g_test_mm256_mul_ps_data.expect;
    int iCount;
    __m256 ma, mb;
    for (iCount = 0; iCount < 2; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m256 res = _mm256_mul_ps(ma, mb);

    return IsEqualFloat32x8(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_mul_epi32()
{
    int32_t *a = g_test_mm512_mul_epi32_data.a;
    int32_t *b = g_test_mm512_mul_epi32_data.b;
    int32_t *expect = g_test_mm512_mul_epi32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < 4; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_mul_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm512_mul_epu32()
{
    uint32_t *a = g_test_mm512_mul_epu32_data.a;
    uint32_t *b = g_test_mm512_mul_epu32_data.b;
    uint32_t *expect = g_test_mm512_mul_epu32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < 4; iCount++) {
        ma.vect_u32[iCount] = vld1q_u32(a + iCount * 4);
        mb.vect_u32[iCount] = vld1q_u32(b + iCount * 4);
    }
    __m512i res = _mm512_mul_epu32(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm512_mul_pd()
{
    float64_t *a = g_test_mm512_mul_pd_data.a;
    float64_t *b = g_test_mm512_mul_pd_data.b;
    float64_t *expect = g_test_mm512_mul_pd_data.expect; 
    int iCount;
    __m512d ma, mb;
    for (iCount = 0; iCount < 4; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m512d res = _mm512_mul_pd(ma, mb);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_mul_ps()
{
    float32_t *a = g_test_mm512_mul_ps_data.a;
    float32_t *b = g_test_mm512_mul_ps_data.b;
    float32_t *expect = g_test_mm512_mul_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < 4; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m512 res = _mm512_mul_ps(ma, mb);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm256_mulhi_epi16()
{
    int16_t *a = g_test_mm256_mulhi_epi16_data.a;
    int16_t *b = g_test_mm256_mulhi_epi16_data.b;
    int16_t *expect = g_test_mm256_mulhi_epi16_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m256i res = _mm256_mulhi_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_mulhi_epu16()
{
    uint16_t *a = g_test_mm256_mulhi_epu16_data.a;
    uint16_t *b = g_test_mm256_mulhi_epu16_data.b;
    uint16_t *expect = g_test_mm256_mulhi_epu16_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_u16[iCount] = vld1q_u16(a + iCount * 8);
        mb.vect_u16[iCount] = vld1q_u16(b + iCount * 8);
    }
    __m256i res = _mm256_mulhi_epu16(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_mulhi_epi16()
{
    int16_t *a = g_test_mm512_mulhi_epi16_data.a;
    int16_t *b = g_test_mm512_mulhi_epi16_data.b;
    int16_t *expect = g_test_mm512_mulhi_epi16_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m512i res = _mm512_mulhi_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_mulhi_epu16()
{
    uint16_t *a = g_test_mm512_mulhi_epu16_data.a;
    uint16_t *b = g_test_mm512_mulhi_epu16_data.b;
    uint16_t *expect = g_test_mm512_mulhi_epu16_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_u16[iCount] = vld1q_u16(a + iCount * 8);
        mb.vect_u16[iCount] = vld1q_u16(b + iCount * 8);
    }
    __m512i res = _mm512_mulhi_epu16(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_mulhi_epi32()
{
    int32_t *a = g_test_mm512_mulhi_epi32_data.a;
    int32_t *b = g_test_mm512_mulhi_epi32_data.b;
    int32_t *expect = g_test_mm512_mulhi_epi32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_mulhi_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_mulhi_epu32()
{
    uint32_t *a = g_test_mm512_mulhi_epu32_data.a;
    uint32_t *b = g_test_mm512_mulhi_epu32_data.b;
    uint32_t *expect = g_test_mm512_mulhi_epu32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_u32[iCount] = vld1q_u32(a + iCount * 4);
        mb.vect_u32[iCount] = vld1q_u32(b + iCount * 4);
    }
    __m512i res = _mm512_mulhi_epu32(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm256_mullo_epi16()
{
    int16_t *a = g_test_mm256_mullo_epi16_data.a;
    int16_t *b = g_test_mm256_mullo_epi16_data.b;
    int16_t *expect = g_test_mm256_mullo_epi16_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m256i res = _mm256_mullo_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_mullo_epi32()
{
    int32_t *a = g_test_mm256_mullo_epi32_data.a;
    int32_t *b = g_test_mm256_mullo_epi32_data.b;
    int32_t *expect = g_test_mm256_mullo_epi32_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_mullo_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_mullo_epi64()
{
    int64_t *a = g_test_mm256_mullo_epi64_data.a;
    int64_t *b = g_test_mm256_mullo_epi64_data.b;
    int64_t *expect = g_test_mm256_mullo_epi64_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    __m256i res = _mm256_mullo_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_mullo_epi16()
{
    int16_t *a = g_test_mm512_mullo_epi16_data.a;
    int16_t *b = g_test_mm512_mullo_epi16_data.b;
    int16_t *expect = g_test_mm512_mullo_epi16_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m512i res = _mm512_mullo_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_mullo_epi32()
{
    int32_t *a = g_test_mm512_mullo_epi32_data.a;
    int32_t *b = g_test_mm512_mullo_epi32_data.b;
    int32_t *expect = g_test_mm512_mullo_epi32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_mullo_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_mullo_epi64()
{
    int64_t *a = g_test_mm512_mullo_epi64_data.a;
    int64_t *b = g_test_mm512_mullo_epi64_data.b;
    int64_t *expect = g_test_mm512_mullo_epi64_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    __m512i res = _mm512_mullo_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_mullox_epi64()
{
    int64_t *a = g_test_mm512_mullox_epi64_data.a;
    int64_t *b = g_test_mm512_mullox_epi64_data.b;
    int64_t *expect = g_test_mm512_mullox_epi64_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    __m512i res = _mm512_mullox_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm256_mulhrs_epi16()
{
    int16_t *a = g_test_mm256_mulhrs_epi16_data.a;
    int16_t *b = g_test_mm256_mulhrs_epi16_data.b;
    int16_t *expect = g_test_mm256_mulhrs_epi16_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m256i res = _mm256_mulhrs_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_mulhrs_epi16()
{
    int16_t *a = g_test_mm512_mulhrs_epi16_data.a;
    int16_t *b = g_test_mm512_mulhrs_epi16_data.b;
    int16_t *expect = g_test_mm512_mulhrs_epi16_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s16[iCount] = vld1q_s16(a + iCount * 8);
        mb.vect_s16[iCount] = vld1q_s16(b + iCount * 8);
    }
    __m512i res = _mm512_mulhrs_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_mul_round_pd()
{
    float64_t *a = g_test_mm512_mul_round_pd_data.a;
    float64_t *b = g_test_mm512_mul_round_pd_data.b;
    float64_t *expect = g_test_mm512_mul_round_pd_data.expect; 
    int iCount;
    __m512d ma, mb;
    for (iCount = 0; iCount < 4; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
        mb.vect_f64[iCount] = vld1q_f64(b + iCount * 2);
    }
    __m512d res = _mm512_mul_round_pd(ma, mb, _MM_FROUND_NO_EXC);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_mul_round_ps()
{
    float32_t *a = g_test_mm512_mul_round_ps_data.a;
    float32_t *b = g_test_mm512_mul_round_ps_data.b;
    float32_t *expect = g_test_mm512_mul_round_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < 4; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
        mb.vect_f32[iCount] = vld1q_f32(b + iCount * 4);
    }
    __m512 res = _mm512_mul_round_ps(ma, mb, _MM_FROUND_NO_EXC);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm_and_si128()
{
    int32_t *a = g_test_mm_and_si128_data.a;
    int32_t *b = g_test_mm_and_si128_data.b;
    int32_t *expect = g_test_mm_and_si128_data.expect;
    __m128i ma, mb;
    ma.vect_s32 = vld1q_s32(a);
    mb.vect_s32 = vld1q_s32(b);
    __m128i res = _mm_and_si128(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm256_and_si256()
{
    int32_t *a = g_test_mm256_and_si256_data.a;
    int32_t *b = g_test_mm256_and_si256_data.b;
    int32_t *expect = g_test_mm256_and_si256_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_and_si256(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm512_and_si512()
{
    int32_t *a = g_test_mm512_and_si512_data.a;
    int32_t *b = g_test_mm512_and_si512_data.b;
    int32_t *expect = g_test_mm512_and_si512_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_and_si512(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm_or_si128()
{
    int32_t *a = g_test_mm_or_si128_data.a;
    int32_t *b = g_test_mm_or_si128_data.b;
    int32_t *expect = g_test_mm_or_si128_data.expect;
    __m128i ma, mb;
    ma.vect_s32 = vld1q_s32(a);
    mb.vect_s32 = vld1q_s32(b);
    __m128i res = _mm_or_si128(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm256_or_si256()
{
    int32_t *a = g_test_mm256_or_si256_data.a;
    int32_t *b = g_test_mm256_or_si256_data.b;
    int32_t *expect = g_test_mm256_or_si256_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_or_si256(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm512_or_si512()
{
    int32_t *a = g_test_mm512_or_si512_data.a;
    int32_t *b = g_test_mm512_or_si512_data.b;
    int32_t *expect = g_test_mm512_or_si512_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_or_si512(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm_andnot_si128()
{
    int32_t *a = g_test_mm_andnot_si128_data.a;
    int32_t *b = g_test_mm_andnot_si128_data.b;
    int32_t *expect = g_test_mm_andnot_si128_data.expect;
    __m128i ma, mb;
    ma.vect_s32 = vld1q_s32(a);
    mb.vect_s32 = vld1q_s32(b);
    __m128i res = _mm_andnot_si128(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm256_andnot_si256()
{
    int32_t *a = g_test_mm256_andnot_si256_data.a;
    int32_t *b = g_test_mm256_andnot_si256_data.b;
    int32_t *expect = g_test_mm256_andnot_si256_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_andnot_si256(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm512_andnot_si512()
{
    int32_t *a = g_test_mm512_andnot_si512_data.a;
    int32_t *b = g_test_mm512_andnot_si512_data.b;
    int32_t *expect = g_test_mm512_andnot_si512_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_andnot_si512(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm_xor_si128()
{
    int32_t *a = g_test_mm_xor_si128_data.a;
    int32_t *b = g_test_mm_xor_si128_data.b;
    int32_t *expect = g_test_mm_xor_si128_data.expect;
    __m128i ma, mb;
    ma.vect_s32 = vld1q_s32(a);
    mb.vect_s32 = vld1q_s32(b);
    __m128i res = _mm_xor_si128(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm256_xor_si256()
{
    int32_t *a = g_test_mm256_xor_si256_data.a;
    int32_t *b = g_test_mm256_xor_si256_data.b;
    int32_t *expect = g_test_mm256_xor_si256_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_xor_si256(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}
int test_mm512_xor_si512()
{
    int32_t *a = g_test_mm512_xor_si512_data.a;
    int32_t *b = g_test_mm512_xor_si512_data.b;
    int32_t *expect = g_test_mm512_xor_si512_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_xor_si512(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm256_or_ps()
{
    uint32_t *a = g_test_mm256_or_ps_data.a;
    uint32_t *b = g_test_mm256_or_ps_data.b;
    uint32_t *expect = g_test_mm256_or_ps_data.expect;
    int iCount;
    __m256 ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vreinterpretq_f32_u32(vld1q_u32(a + iCount * 4));
        mb.vect_f32[iCount] = vreinterpretq_f32_u32(vld1q_u32(b + iCount * 4));
    }
    __m256 res = _mm256_or_ps(ma, mb);
    
    return comp_return(expect, &res, sizeof(__m256));
}

int test_mm256_or_pd()
{
    uint64_t *a = g_test_mm256_or_pd_data.a;
    uint64_t *b = g_test_mm256_or_pd_data.b;
    uint64_t *expect = g_test_mm256_or_pd_data.expect;
    int iCount;
    union {
        __m256d f;
        __m256i i;
    }ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.i.vect_u64[iCount] = vld1q_u64(a + iCount * 2);
        mb.i.vect_u64[iCount] = vld1q_u64(b + iCount * 2);
    }
    __m256d res = _mm256_or_pd(ma.f, mb.f);

    return comp_return(expect, &res, sizeof(__m256d));
}

int test_mm512_and_epi32()
{
    int32_t *a = g_test_mm512_and_epi32_data.a;
    int32_t *b = g_test_mm512_and_epi32_data.b;
    int32_t *expect = g_test_mm512_and_epi32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_and_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_and_epi64()
{
    int64_t *a = g_test_mm512_and_epi64_data.a;
    int64_t *b = g_test_mm512_and_epi64_data.b;
    int64_t *expect = g_test_mm512_and_epi64_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    __m512i res = _mm512_and_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_or_epi32()
{
    int32_t *a = g_test_mm512_or_epi32_data.a;
    int32_t *b = g_test_mm512_or_epi32_data.b;
    int32_t *expect = g_test_mm512_or_epi32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_or_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_or_epi64()
{
    int64_t *a = g_test_mm512_or_epi64_data.a;
    int64_t *b = g_test_mm512_or_epi64_data.b;
    int64_t *expect = g_test_mm512_or_epi64_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    __m512i res = _mm512_or_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_xor_ps()
{
    uint32_t *a = g_test_mm512_xor_ps_data.a;
    uint32_t *b = g_test_mm512_xor_ps_data.b;
    uint32_t *expect = g_test_mm512_xor_ps_data.expect;
    int iCount;
    __m512 ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vreinterpretq_f32_u32(vld1q_u32(a + iCount * 4));
        mb.vect_f32[iCount] = vreinterpretq_f32_u32(vld1q_u32(b + iCount * 4));
    }
    __m512 res = _mm512_xor_ps(ma, mb);

    return comp_return(expect, &res, sizeof(__m512));
}

int test_mm512_xor_pd()
{
    uint64_t *a = g_test_mm512_xor_pd_data.a;
    uint64_t *b = g_test_mm512_xor_pd_data.b;
    uint64_t *expect = g_test_mm512_xor_pd_data.expect;
    int iCount;
    union {
        __m512d f;
        __m512i i;
    }ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.i.vect_u64[iCount] = vld1q_u64(a + iCount * 2);
        mb.i.vect_u64[iCount] = vld1q_u64(b + iCount * 2);
    }
    __m512d res = _mm512_xor_pd(ma.f, mb.f);

    return comp_return(expect, &res, sizeof(__m512d));
}

int test_mm256_cmpeq_epi8()
{
    int8_t *a = g_test_mm256_cmpeq_epi8_data.a;
    int8_t *b = g_test_mm256_cmpeq_epi8_data.b;
    uint8_t *expect = g_test_mm256_cmpeq_epi8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m256i res = _mm256_cmpeq_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_cmpeq_epi32()
{
    int32_t *a = g_test_mm256_cmpeq_epi32_data.a;
    int32_t *b = g_test_mm256_cmpeq_epi32_data.b;
    uint32_t *expect = g_test_mm256_cmpeq_epi32_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_cmpeq_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_cmpgt_epi32()
{
    int32_t *a = g_test_mm256_cmpgt_epi32_data.a;
    int32_t *b = g_test_mm256_cmpgt_epi32_data.b;
    uint32_t *expect = g_test_mm256_cmpgt_epi32_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_cmpgt_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm_cmpeq_epi8()
{
    int8_t *a = g_test_mm_cmpeq_epi8_data.a;
    int8_t *b = g_test_mm_cmpeq_epi8_data.b;
    uint8_t *expect = g_test_mm_cmpeq_epi8_data.expect;
    __m128i ma, mb;
    ma.vect_s8 = vld1q_s8(a);
    mb.vect_s8 = vld1q_s8(b);
    __m128i res = _mm_cmpeq_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_cmpeq_epi32()
{
    int32_t *a = g_test_mm_cmpeq_epi32_data.a;
    int32_t *b = g_test_mm_cmpeq_epi32_data.b;
    uint32_t *expect = g_test_mm_cmpeq_epi32_data.expect;
    __m128i ma, mb;
    ma.vect_s32 = vld1q_s32(a);
    mb.vect_s32 = vld1q_s32(b);
    __m128i res = _mm_cmpeq_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_cmpeq_epi64()
{
    int64_t *a = g_test_mm_cmpeq_epi64_data.a;
    int64_t *b = g_test_mm_cmpeq_epi64_data.b;
    uint64_t *expect = g_test_mm_cmpeq_epi64_data.expect;
    __m128i ma, mb;
    ma.vect_s64 = vld1q_s64(a);
    mb.vect_s64 = vld1q_s64(b);
    __m128i res = _mm_cmpeq_epi64(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm512_cmp_epi32_mask()
{
    int32_t *a = g_test_mm512_cmp_epi32_mask_data.a;
    int32_t *b = g_test_mm512_cmp_epi32_mask_data.b;
    __mmask16 *expect = g_test_mm512_cmp_epi32_mask_data.expect;
    __m512i ma, mb;
    int iCount;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __mmask16 res[8];
    res[0] = _mm512_cmp_epi32_mask(ma, mb, _MM_CMPINT_EQ);
    res[1] = _mm512_cmp_epi32_mask(ma, mb, _MM_CMPINT_LT);
    res[2] = _mm512_cmp_epi32_mask(ma, mb, _MM_CMPINT_LE);
    res[3] = _mm512_cmp_epi32_mask(ma, mb, _MM_CMPINT_FALSE);
    res[4] = _mm512_cmp_epi32_mask(ma, mb, _MM_CMPINT_NE);
    res[5] = _mm512_cmp_epi32_mask(ma, mb, _MM_CMPINT_NLT);
    res[6] = _mm512_cmp_epi32_mask(ma, mb, _MM_CMPINT_NLE);
    res[7] = _mm512_cmp_epi32_mask(ma, mb, _MM_CMPINT_TRUE);
    return comp_return(expect, res, 8 * sizeof(__mmask16));
}

int test_mm512_cmp_epi8_mask()
{
    int8_t *a = g_test_mm512_cmp_epi8_mask_data.a;
    int8_t *b = g_test_mm512_cmp_epi8_mask_data.b;
    __mmask64 *expect = g_test_mm512_cmp_epi8_mask_data.expect;
    __m512i ma, mb;
    int iCount;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __mmask64 res[8];
    res[0] = _mm512_cmp_epi8_mask(ma, mb, _MM_CMPINT_EQ);
    res[1] = _mm512_cmp_epi8_mask(ma, mb, _MM_CMPINT_LT);
    res[2] = _mm512_cmp_epi8_mask(ma, mb, _MM_CMPINT_LE);
    res[3] = _mm512_cmp_epi8_mask(ma, mb, _MM_CMPINT_FALSE);
    res[4] = _mm512_cmp_epi8_mask(ma, mb, _MM_CMPINT_NE);
    res[5] = _mm512_cmp_epi8_mask(ma, mb, _MM_CMPINT_NLT);
    res[6] = _mm512_cmp_epi8_mask(ma, mb, _MM_CMPINT_NLE);
    res[7] = _mm512_cmp_epi8_mask(ma, mb, _MM_CMPINT_TRUE);
    return comp_return(expect, res, 8 * sizeof(__mmask64));
}

int test_mm512_cmpeq_epi8_mask()
{
    int8_t *a = g_test_mm512_cmpeq_epi8_mask_data.a;
    int8_t *b = g_test_mm512_cmpeq_epi8_mask_data.b;
    __mmask64 expect = g_test_mm512_cmpeq_epi8_mask_data.expect;
    __m512i ma, mb;
    int iCount;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __mmask64 res;
    res = _mm512_cmpeq_epi8_mask(ma, mb);
    return res == expect;
}

int test_mm512_cmpgt_epi32_mask()
{
    int32_t *a = g_test_mm512_cmpgt_epi32_mask_data.a;
    int32_t *b = g_test_mm512_cmpgt_epi32_mask_data.b;
    __mmask16 expect = g_test_mm512_cmpgt_epi32_mask_data.expect;
    __m512i ma, mb;
    int iCount;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __mmask16 res;
    res = _mm512_cmpgt_epi32_mask(ma, mb);
    return res == expect;
}

int test_mm512_mask_cmpeq_epi8_mask()
{
    int8_t *a = g_test_mm512_mask_cmpeq_epi8_mask_data.a;
    int8_t *b = g_test_mm512_mask_cmpeq_epi8_mask_data.b;
    __mmask64 k1 = g_test_mm512_mask_cmpeq_epi8_mask_data.k1;
    __mmask64 expect = g_test_mm512_mask_cmpeq_epi8_mask_data.expect;
    __m512i ma, mb;
    int iCount;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __mmask64 res;
    res = _mm512_mask_cmpeq_epi8_mask(k1, ma, mb);
    return res == expect;
}

int test_mm512_set_epi32()
{
    int32_t *a = g_test_mm512_set_epi32_data.a;
    int32_t *expect = g_test_mm512_set_epi32_data.expect;
    __m512i res = _mm512_set_epi32(a[15], a[14], a[13], a[12], a[11], a[10], a[9], a[8], a[7], a[6], a[5], a[4], a[3],
                                   a[2], a[1], a[0]);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_set_epi64()
{
    int64_t *a = g_test_mm512_set_epi64_data.a;
    int64_t *expect = g_test_mm512_set_epi64_data.expect;
    __m512i res = _mm512_set_epi64(a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0]);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm512_set1_epi32()
{
    int32_t a = g_test_mm512_set1_epi32_data.a;
    int32_t *expect = g_test_mm512_set1_epi32_data.expect;
    __m512i res = _mm512_set1_epi32(a);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm512_set1_epi64()
{
    int64_t a = g_test_mm512_set1_epi64_data.a;
    int64_t *expect = g_test_mm512_set1_epi64_data.expect;
    __m512i res = _mm512_set1_epi64(a);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm512_set1_epi8()
{
    int8_t a = g_test_mm512_set1_epi8_data.a;
    int8_t *expect = g_test_mm512_set1_epi8_data.expect;
    __m512i res = _mm512_set1_epi8((char)a);
    return comp_return(expect, &res, sizeof(__m512i));
}
int test_mm512_set_ps()
{
    float32_t *a = g_test_mm512_set_ps_data.a;
    float32_t *expect = g_test_mm512_set_ps_data.expect;
    __m512 res = _mm512_set_ps(a[15], a[14], a[13], a[12], a[11], a[10], a[9], a[8], a[7], a[6], a[5], a[4], a[3], a[2],
                               a[1], a[0]);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}
int test_mm512_set_pd()
{
    float64_t *a = g_test_mm512_set_pd_data.a;
    float64_t *expect = g_test_mm512_set_pd_data.expect;
    __m512d res = _mm512_set_pd(a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0]);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}
int test_mm512_set1_ps()
{
    float32_t a = g_test_mm512_set1_ps_data.a;
    float32_t *expect = g_test_mm512_set1_ps_data.expect;
    __m512 res = _mm512_set1_ps(a);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}
int test_mm512_set1_pd()
{
    float64_t a = g_test_mm512_set1_pd_data.a;
    float64_t *expect = g_test_mm512_set1_pd_data.expect;
    __m512d res = _mm512_set1_pd(a);

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}
int test_mm512_setzero_ps()
{
    float32_t *expect = g_test_mm512_setzero_ps_data.expect;
    __m512 res = _mm512_setzero_ps();

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}
int test_mm512_setzero_pd()
{
    float64_t *expect = g_test_mm512_setzero_pd_data.expect;
    __m512d res = _mm512_setzero_pd();

    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm_move_sd()
{
    float64_t *a = g_test_mm_move_sd_data.a;
    float64_t *b = g_test_mm_move_sd_data.b;
    float64_t *expect = g_test_mm_move_sd_data.expect;
    __m128d ma, mb, res;
    ma = vld1q_f64(a);
    mb = vld1q_f64(b);
    res = _mm_move_sd(ma, mb);

    return IsEqualFloat64x2(res, expect, DEFAULT_EPSILON_F64);
}
int test_mm_move_ss()
{
    float32_t *a = g_test_mm_move_ss_data.a;
    float32_t *b = g_test_mm_move_ss_data.b;
    float32_t *expect = g_test_mm_move_ss_data.expect;
    __m128 ma, mb, res;
    ma = vld1q_f32(a);
    mb = vld1q_f32(b);
    res = _mm_move_ss(ma, mb);

    return IsEqualFloat32x4(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm_movemask_epi8()
{
    int8_t *a = g_test_mm_movemask_epi8_data.a;
    int expect = g_test_mm_movemask_epi8_data.expect;
    __m128i ma;
    ma.vect_s8 = vld1q_s8(a);
    int res = _mm_movemask_epi8(ma);

    return res == expect;
}
int test_mm_movemask_ps()
{
    float32_t *a = g_test_mm_movemask_ps_data.a;
    int expect = g_test_mm_movemask_ps_data.expect;
    __m128 ma;
    ma = vld1q_f32(a);
    int res = _mm_movemask_ps(ma);

    return res == expect;
}

int test_mm256_movemask_epi8()
{
    int8_t *a = g_test_mm256_movemask_epi8_data.a;
    int expect = g_test_mm256_movemask_epi8_data.expect;
    int iCount;
    __m256i ma;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
    }
    int res = _mm256_movemask_epi8(ma);

    return res == expect;
}
int test_mm256_movemask_ps()
{
    float32_t *a = g_test_mm256_movemask_ps_data.a;
    int expect = g_test_mm256_movemask_ps_data.expect;
    int iCount;
    __m256 ma;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
    }
    int res = _mm256_movemask_ps(ma);

    return res == expect;
}

int test_mm_testz_si128()
{
    int8_t *a = g_test_mm_testz_si128_data.a;
    int8_t *b = g_test_mm_testz_si128_data.b;
    int expect = g_test_mm_testz_si128_data.expect;
    __m128i ma, mb;
    ma.vect_s8 = vld1q_s8(a);
    mb.vect_s8 = vld1q_s8(b);
    int res = _mm_testz_si128(ma, mb);

    return res == expect;
}
int test_mm256_testz_si256()
{
    int8_t *a = g_test_mm256_testz_si256_data.a;
    int8_t *b = g_test_mm256_testz_si256_data.b;
    int expect = g_test_mm256_testz_si256_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    int res = _mm256_testz_si256(ma, mb);
    return res == expect;
}

int test_mm512_movm_epi8()
{
    __mmask64 a = g_test_mm512_movm_epi8_data.a;
    int8_t *expect = g_test_mm512_movm_epi8_data.expect;
    __m512i res = _mm512_movm_epi8(a);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_movm_epi32()
{
    __mmask16 a = g_test_mm512_movm_epi32_data.a;
    int32_t *expect = g_test_mm512_movm_epi32_data.expect;
    __m512i res = _mm512_movm_epi32(a);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm_extract_epi32()
{
    int32_t *a = g_test_mm_extract_epi32_data.a;
    const int b = g_test_mm_extract_epi32_data.b;
    int32_t expect = g_test_mm_extract_epi32_data.expect;
    __m128i ma;
    ma.vect_s32 = vld1q_s32(a);

    int res = _mm_extract_epi32(ma, b);
    return expect == res;
}
int test_mm_extract_epi64()
{
    int64_t *a = g_test_mm_extract_epi64_data.a;
    const int b = g_test_mm_extract_epi64_data.b;
    int64_t expect = g_test_mm_extract_epi64_data.expect;
    __m128i ma;
    ma.vect_s64 = vld1q_s64(a);

    int64_t res = _mm_extract_epi64(ma, b);
    return expect == res;
}
int test_mm256_extracti128_si256()
{
    int64_t *a = g_test_mm256_extracti128_si256_data.a;
    const int b = g_test_mm256_extracti128_si256_data.b;
    int64_t *expect = g_test_mm256_extracti128_si256_data.expect;

    int iCount;
    __m256i ma;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }

    __m128i res = _mm256_extracti128_si256(ma, b);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_extract_ps()
{
    float32_t *a = g_test_mm_extract_ps_data.a;
    const int b = g_test_mm_extract_ps_data.b;
    int expect = g_test_mm_extract_ps_data.expect;
    __m128 ma;
    ma = vld1q_f32(a);
    int res = _mm_extract_ps(ma, b);

    return expect == res;
}

int test_mm256_extract_epi32()
{
    int32_t *a = g_test_mm256_extract_epi32_data.a;
    const int b = g_test_mm256_extract_epi32_data.b;
    int32_t expect = g_test_mm256_extract_epi32_data.expect;
    __m256i ma;
    int iCount;
    for (iCount = 0; iCount < g_256bit_divto_128bit; ++iCount) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    int32_t res = _mm256_extract_epi32(ma, b);

    return res == expect;
}
int test_mm256_extract_epi64()
{
    int64_t *a = g_test_mm256_extract_epi64_data.a;
    const int b = g_test_mm256_extract_epi64_data.b;
    int64_t expect = g_test_mm256_extract_epi64_data.expect;
    __m256i ma;
    int iCount;
    for (iCount = 0; iCount < g_256bit_divto_128bit; ++iCount) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    int64_t res = _mm256_extract_epi64(ma, b);

    return res == expect;
}
int test_mm256_extractf128_ps()
{
    float32_t *a = g_test_mm256_extractf128_ps_data.a;
    const int b = g_test_mm256_extractf128_ps_data.b;
    float32_t *expect = g_test_mm256_extractf128_ps_data.expect;
    __m256 ma;
    int iCount;
    for (iCount = 0; iCount < g_256bit_divto_128bit; ++iCount) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
    }
    __m128 res = _mm256_extractf128_ps(ma, b);

    return IsEqualFloat32x4(res, expect, DEFAULT_EPSILON_F32);
}
int test_mm256_extractf128_pd()
{
    float64_t *a = g_test_mm256_extractf128_pd_data.a;
    const int b = g_test_mm256_extractf128_pd_data.b;
    float64_t *expect = g_test_mm256_extractf128_pd_data.expect;
    __m256d ma;
    int iCount;
    for (iCount = 0; iCount < g_256bit_divto_128bit; ++iCount) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
    }
    __m128d res = _mm256_extractf128_pd(ma, b);

    return IsEqualFloat64x2(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_extractf32x8_ps()
{
    float32_t *a = g_test_mm512_extractf32x8_ps_data.a;
    const int b = g_test_mm512_extractf32x8_ps_data.b;
    float32_t *expect = g_test_mm512_extractf32x8_ps_data.expect;
    __m512 ma;
    int iCount;
    for (iCount = 0; iCount < g_512bit_divto_128bit; ++iCount) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
    }
    __m256 res = _mm512_extractf32x8_ps(ma, b);

    return IsEqualFloat32x8(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_extractf64x4_pd()
{
    float64_t *a = g_test_mm512_extractf64x4_pd_data.a;
    const int b = g_test_mm512_extractf64x4_pd_data.b;
    float64_t *expect = g_test_mm512_extractf64x4_pd_data.expect;
    __m512d ma;
    int iCount;
    for (iCount = 0; iCount < g_512bit_divto_128bit; ++iCount) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
    }
    __m256d res = _mm512_extractf64x4_pd(ma, b);

    return IsEqualFloat64x4(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm_crc32_u8()
{
    unsigned int crc = g_test_mm_crc32_u8_data.crc;
    unsigned char v = g_test_mm_crc32_u8_data.v;
    unsigned int expect = g_test_mm_crc32_u8_data.expect;
    unsigned int res = _mm_crc32_u8(crc, v);
    return res == expect;
}
int test_mm_crc32_u16()
{
    unsigned int crc = g_test_mm_crc32_u16_data.crc;
    unsigned short v = g_test_mm_crc32_u16_data.v;
    unsigned int expect = g_test_mm_crc32_u16_data.expect;
    unsigned int res = _mm_crc32_u16(crc, v);
    return res == expect;
}
int test_mm_crc32_u32()
{
    unsigned int crc = g_test_mm_crc32_u32_data.crc;
    unsigned int v = g_test_mm_crc32_u32_data.v;
    unsigned int expect = g_test_mm_crc32_u32_data.expect;
    unsigned int res = _mm_crc32_u32(crc, v);
    return res == expect;
}
int test_mm_crc32_u64()
{
    unsigned __int64 crc = g_test_mm_crc32_u64_data.crc;
    unsigned __int64 v = g_test_mm_crc32_u64_data.v;
    unsigned __int64 expect = g_test_mm_crc32_u64_data.expect;
    unsigned __int64 res = _mm_crc32_u64(crc, v);
    return res == expect;
}

int test_mm_shuffle_epi8()
{
    int8_t *a = g_test_mm_shuffle_epi8_data.a;
    int8_t *b = g_test_mm_shuffle_epi8_data.b;
    int8_t *expect = g_test_mm_shuffle_epi8_data.expect;
    __m128i ma, mb;
    ma.vect_s8 = vld1q_s8(a);
    mb.vect_s8 = vld1q_s8(b);
    __m128i res = _mm_shuffle_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm256_shuffle_epi8()
{
    int8_t *a = g_test_mm256_shuffle_epi8_data.a;
    int8_t *b = g_test_mm256_shuffle_epi8_data.b;
    int8_t *expect = g_test_mm256_shuffle_epi8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m256i res = _mm256_shuffle_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_shuffle_epi8()
{
    int8_t *a = g_test_mm512_shuffle_epi8_data.a;
    int8_t *b = g_test_mm512_shuffle_epi8_data.b;
    int8_t *expect = g_test_mm512_shuffle_epi8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m512i res = _mm512_shuffle_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_maskz_shuffle_epi8()
{
    int8_t *a = g_test_mm512_maskz_shuffle_epi8_data.a;
    int8_t *b = g_test_mm512_maskz_shuffle_epi8_data.b;
    __mmask64 k = g_test_mm512_maskz_shuffle_epi8_data.k;
    int8_t *expect = g_test_mm512_maskz_shuffle_epi8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m512i res = _mm512_maskz_shuffle_epi8(k, ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm256_multishift_epi64_epi8()
{
    uint8_t *a = g_test_mm256_multishift_epi64_epi8_data.a;
    uint8_t *b = g_test_mm256_multishift_epi64_epi8_data.b;
    uint8_t *expect = g_test_mm256_multishift_epi64_epi8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_u8[iCount] = vld1q_u8(a + iCount * 16);
        mb.vect_u8[iCount] = vld1q_u8(b + iCount * 16);
    }
    __m256i res = _mm256_multishift_epi64_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_multishift_epi64_epi8()
{
    uint8_t *a = g_test_mm512_multishift_epi64_epi8_data.a;
    uint8_t *b = g_test_mm512_multishift_epi64_epi8_data.b;
    uint8_t *expect = g_test_mm512_multishift_epi64_epi8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_u8[iCount] = vld1q_u8(a + iCount * 16);
        mb.vect_u8[iCount] = vld1q_u8(b + iCount * 16);
    }
    __m512i res = _mm512_multishift_epi64_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm256_unpacklo_epi8()
{
    int8_t *a = g_test_mm256_unpacklo_epi8_data.a;
    int8_t *b = g_test_mm256_unpacklo_epi8_data.b;
    int8_t *expect = g_test_mm256_unpacklo_epi8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m256i res = _mm256_unpacklo_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_unpackhi_epi8()
{
    int8_t *a = g_test_mm256_unpackhi_epi8_data.a;
    int8_t *b = g_test_mm256_unpackhi_epi8_data.b;
    int8_t *expect = g_test_mm256_unpackhi_epi8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m256i res = _mm256_unpackhi_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_unpacklo_epi8()
{
    int8_t *a = g_test_mm512_unpacklo_epi8_data.a;
    int8_t *b = g_test_mm512_unpacklo_epi8_data.b;
    int8_t *expect = g_test_mm512_unpacklo_epi8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m512i res = _mm512_unpacklo_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_unpackhi_epi8()
{
    int8_t *a = g_test_mm512_unpackhi_epi8_data.a;
    int8_t *b = g_test_mm512_unpackhi_epi8_data.b;
    int8_t *expect = g_test_mm512_unpackhi_epi8_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m512i res = _mm512_unpackhi_epi8(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm_set_pd()
{
    float64_t *src = g_test_mm_set_pd_data.a;
    __m128d dst = _mm_set_pd(src[1], src[0]);

    return comp_return(g_test_mm_set_pd_data.expect, &dst, sizeof(dst));
}

int test_mm256_set_epi32()
{
    int32_t *src = g_test_mm256_set_epi32_data.a;
    __m256i dst = _mm256_set_epi32(src[7], src[6], src[5], src[4], src[3], src[2], src[1], src[0]);

    return comp_return(g_test_mm256_set_epi32_data.expect, &dst, sizeof(dst));
}

int test_mm256_set_epi64x()
{
    int64_t *src = g_test_mm256_set_epi64x_data.a;
    __m256i dst = _mm256_set_epi64x(src[3], src[2], src[1], src[0]);

    return comp_return(g_test_mm256_set_epi64x_data.expect, &dst, sizeof(dst));
}

int test_mm256_set_m128i()
{
    int32_t *src = g_test_mm256_set_m128i_data.a;
    __m128i low = _mm_set_epi32(src[3], src[2], src[1], src[0]);
    __m128i high = _mm_set_epi32(src[7], src[6], src[5], src[4]);
    __m256i dst = _mm256_set_m128i(high, low);

    return comp_return(g_test_mm256_set_m128i_data.expect, &dst, sizeof(dst));
}

int test_mm256_set_ps()
{
    float32_t *src = g_test_mm256_set_ps_data.a;
    __m256 dst = _mm256_set_ps(src[7], src[6], src[5], src[4], src[3], src[2], src[1], src[0]);

    return comp_return(g_test_mm256_set_ps_data.expect, &dst, sizeof(dst));
}

int test_mm256_set_pd()
{
    float64_t *src = g_test_mm256_set_pd_data.a;
    __m256d dst = _mm256_set_pd(src[3], src[2], src[1], src[0]);

    return comp_return(g_test_mm256_set_pd_data.expect, &dst, sizeof(dst));
}

int test_mm_setzero_si128()
{
    __m128i dst = _mm_setzero_si128();

    return comp_return(g_test_mm_setzero_si128_data.expect, &dst, sizeof(dst));
}

int test_mm256_setzero_si256()
{
    __m256i dst = _mm256_setzero_si256();

    return comp_return(g_test_mm256_setzero_si256_data.expect, &dst, sizeof(dst));
}

int test_mm512_setzero_si512()
{
    __m512i dst = _mm512_setzero_si512();

    return comp_return(g_test_mm512_setzero_si512_data.expect, &dst, sizeof(dst));
}

int test_mm256_setzero_ps()
{
    __m256 dst = _mm256_setzero_ps();

    return comp_return(g_test_mm256_setzero_ps_data.expect, &dst, sizeof(dst));
}

int test_mm256_setzero_pd()
{
    __m256d dst = _mm256_setzero_pd();

    return comp_return(g_test_mm256_setzero_pd_data.expect, &dst, sizeof(dst));
}

int test_mm_set1_epi8()
{
    __m128i dst = _mm_set1_epi8(g_test_mm_set1_epi8_data.a);

    return comp_return(g_test_mm_set1_epi8_data.expect, &dst, sizeof(dst));
}

int test_mm_set1_epi32()
{
    __m128i dst = _mm_set1_epi32(g_test_mm_set1_epi32_data.a);

    return comp_return(g_test_mm_set1_epi32_data.expect, &dst, sizeof(dst));
}

int test_mm_set1_ps()
{
    __m128 dst = _mm_set1_ps(g_test_mm_set1_ps_data.a);

    return comp_return(g_test_mm_set1_ps_data.expect, &dst, sizeof(dst));
}

int test_mm_set1_epi64x()
{
    __m128i dst = _mm_set1_epi64x(g_test_mm_set1_epi64x_data.a);

    return comp_return(g_test_mm_set1_epi64x_data.expect, &dst, sizeof(dst));
}

int test_mm_set1_pd()
{
    __m128d dst = _mm_set1_pd(g_test_mm_set1_pd_data.a);

    return comp_return(g_test_mm_set1_pd_data.expect, &dst, sizeof(dst));
}

int test_mm256_set1_epi8()
{
    __m256i dst = _mm256_set1_epi8(g_test_mm256_set1_epi8_data.a);

    return comp_return(g_test_mm256_set1_epi8_data.expect, &dst, sizeof(dst));
}

int test_mm256_set1_epi32()
{
    __m256i dst = _mm256_set1_epi32(g_test_mm256_set1_epi32_data.a);

    return comp_return(g_test_mm256_set1_epi32_data.expect, &dst, sizeof(dst));
}

int test_mm256_set1_epi64x()
{
    __m256i dst = _mm256_set1_epi64x(g_test_mm256_set1_epi64x_data.a);

    return comp_return(g_test_mm256_set1_epi64x_data.expect, &dst, sizeof(dst));
}

int test_mm256_set1_pd()
{
    __m256d dst = _mm256_set1_pd(g_test_mm256_set1_pd_data.a);

    return comp_return(g_test_mm256_set1_pd_data.expect, &dst, sizeof(dst));
}

int test_mm256_set1_ps()
{
    __m256 dst = _mm256_set1_ps(g_test_mm256_set1_ps_data.a);

    return comp_return(g_test_mm256_set1_ps_data.expect, &dst, sizeof(dst));
}

int test_mm_loadu_si128()
{
    int32_t *src = g_test_mm_loadu_si128_data.a;
    __m128i data = _mm_set_epi32(src[3], src[2], src[1], src[0]);
    __m128i dst = _mm_loadu_si128(&data);

    return comp_return(g_test_mm_loadu_si128_data.expect, &dst, sizeof(dst));
}

int test_mm256_load_si256()
{
    int32_t *src = g_test_mm256_load_si256_data.a;
    __m256i data = _mm256_set_epi32(src[7], src[6], src[5], src[4], src[3], src[2], src[1], src[0]);
    __m256i dst = _mm256_load_si256(&data);

    return comp_return(g_test_mm256_load_si256_data.expect, &dst, sizeof(dst));
}

int test_mm256_loadu_si256()
{
    int32_t *src = g_test_mm256_loadu_si256_data.a;
    __m256i data = _mm256_set_epi32(src[7], src[6], src[5], src[4], src[3], src[2], src[1], src[0]);
    __m256i dst = _mm256_loadu_si256(&data);

    return comp_return(g_test_mm256_loadu_si256_data.expect, &dst, sizeof(dst));
}

int test_mm256_maskload_epi32()
{
    int32_t *src = g_test_mm256_maskload_epi32_data.a;
    int32_t *maskSrc = g_test_mm256_maskload_epi32_data.mask;

    __m256i mask = _mm256_set_epi32(maskSrc[7], maskSrc[6], maskSrc[5], maskSrc[4], maskSrc[3], maskSrc[2], maskSrc[1],
                                    maskSrc[0]);
    __m256i dst = _mm256_maskload_epi32(src, mask);

    return comp_return(g_test_mm256_maskload_epi32_data.expect, &dst, sizeof(dst));
}

int test_mm512_load_si512()
{
    int32_t *src = g_test_mm512_load_si512_data.a;
    __m512i dst;

    dst = _mm512_load_si512(src);

    return comp_return(g_test_mm512_load_si512_data.expect, &dst, sizeof(dst));
}

int test_mm512_loadu_si512()
{
    __m512i dst = _mm512_loadu_si512(g_test_mm512_loadu_si512_data.a);

    return comp_return(g_test_mm512_loadu_si512_data.expect, &dst, sizeof(dst));
}

int test_mm512_mask_loadu_epi8()
{
    __m512i src = _mm512_set1_epi8(g_test_mm512_mask_loadu_epi8_data.src);
    int8_t *mem_addr = g_test_mm512_mask_loadu_epi8_data.mem_addr;
    unsigned long long mask = g_test_mm512_mask_loadu_epi8_data.mask;
    __m512i dst = _mm512_mask_loadu_epi8(src, mask, mem_addr);

    return comp_return(g_test_mm512_mask_loadu_epi8_data.expect, &dst, sizeof(dst));
}

int test_mm512_maskz_loadu_epi8()
{
    int8_t *mem_addr = g_test_mm512_maskz_loadu_epi8_data.mem_addr;
    unsigned long long k = g_test_mm512_maskz_loadu_epi8_data.k;
    __m512i dst = _mm512_maskz_loadu_epi8(k, mem_addr);

    return comp_return(g_test_mm512_maskz_loadu_epi8_data.expect, &dst, sizeof(dst));
}

int test_mm512_abs_epi8()
{
    __m512i a;
    __m512i dst;

    a = _mm512_loadu_si512(g_test_mm512_abs_epi8_data.a);
    dst = _mm512_abs_epi8(a);

    return comp_return(g_test_mm512_abs_epi8_data.expect, &dst, sizeof(dst));
}

int test_mm256_broadcastq_epi64()
{
    int64_t *src = g_test_mm256_broadcastq_epi64_data.a;
    __m128i a = _mm_set_epi64x(src[1], src[0]);
    __m256i dst = _mm256_broadcastq_epi64(a);

    return comp_return(g_test_mm256_broadcastq_epi64_data.expect, &dst, sizeof(dst));
}

int test_mm256_broadcastsi128_si256()
{
    int64_t *src = g_test_mm256_broadcastsi128_si256_data.a;

    __m128i a = _mm_set_epi64x(src[1], src[0]);
    __m256i dst = _mm256_broadcastsi128_si256(a);

    return comp_return(g_test_mm256_broadcastsi128_si256_data.expect, &dst, sizeof(dst));
}

int test_mm512_broadcast_i32x4()
{
    int32_t *src = g_test_mm512_broadcast_i32x4_data.a;

    __m128i a = _mm_set_epi32(src[3], src[2], src[1], src[0]);
    __m512i dst = _mm512_broadcast_i32x4(a);

    return comp_return(g_test_mm512_broadcast_i32x4_data.expect, &dst, sizeof(dst));
}

int test_mm512_broadcast_i64x4()
{
    int64_t *src = g_test_mm512_broadcast_i64x4_data.a;

    __m256i a = _mm256_set_epi64x(src[3], src[2], src[1], src[0]);
    __m512i dst = _mm512_broadcast_i64x4(a);

    return comp_return(g_test_mm512_broadcast_i64x4_data.expect, &dst, sizeof(dst));
}

int test_mm512_mask_broadcast_i64x4()
{
    __mmask8 k = g_test_mm512_mask_broadcast_i64x4_data.k;
    __m512i src = _mm512_loadu_si512(g_test_mm512_mask_broadcast_i64x4_data.src);
    int64_t *addr = g_test_mm512_mask_broadcast_i64x4_data.a;
    __m256i a = _mm256_set_epi64x(addr[3], addr[2], addr[1], addr[0]);
    __m512i dst = _mm512_mask_broadcast_i64x4(src, k, a);

    return comp_return(g_test_mm512_mask_broadcast_i64x4_data.expect, &dst, sizeof(dst));
}

int test_mm256_castpd128_pd256()
{
    __m128d a = vld1q_f64(g_test_mm256_castpd128_pd256_data.a);
    __m256d dst = _mm256_castpd128_pd256(a);

    return IsEqualFloat64x2(dst.vect_f64[0], g_test_mm256_castpd128_pd256_data.expect, DEFAULT_EPSILON_F64);
}

int test_mm256_castpd256_pd128()
{
    float64_t *src = g_test_mm256_castpd256_pd128_data.a;
    __m256d a = _mm256_set_pd(src[3], src[2], src[1], src[0]);
    __m128d dst = _mm256_castpd256_pd128(a);

    return comp_return(g_test_mm256_castpd256_pd128_data.expect, &dst, sizeof(dst));
}

int test_mm256_castps128_ps256()
{
    __m128 a = vld1q_f32(g_test_mm256_castps128_ps256_data.a);
    __m256 dst = _mm256_castps128_ps256(a);

    return IsEqualFloat32x4(dst.vect_f32[0], g_test_mm256_castps128_ps256_data.expect, DEFAULT_EPSILON_F32);
}

int test_mm256_castps256_ps128()
{
    float32_t *src = g_test_mm256_castps256_ps128_data.a;
    __m256 a = _mm256_set_ps(src[7], src[6], src[5], src[4], src[3], src[2], src[1], src[0]);

    __m128 dst = _mm256_castps256_ps128(a);

    return comp_return(g_test_mm256_castps256_ps128_data.expect, &dst, sizeof(dst));
}

int test_mm_castsi128_ps()
{
    int32_t *src = g_test_mm_castsi128_ps_data.a;

    __m128i a = _mm_set_epi32(src[3], src[2], src[1], src[0]);
    __m128 dst = _mm_castsi128_ps(a);

    return comp_return(g_test_mm_castsi128_ps_data.expect, &dst, sizeof(dst));
}

int test_mm256_castsi128_si256()
{
    int32_t *src = g_test_mm256_castsi128_si256_data.a;
    __m128i a = _mm_set_epi32(src[3], src[2], src[1], src[0]);
    __m256i dst = _mm256_castsi128_si256(a);

    return comp_return(g_test_mm256_castsi128_si256_data.expect, &dst, sizeof(__m128i));
}

int test_mm256_castsi256_ps()
{
    int32_t *src = g_test_mm256_castsi256_ps_data.a;

    __m256i a = _mm256_set_epi32(src[7], src[6], src[5], src[4], src[3], src[2], src[1], src[0]);
    __m256 dst = _mm256_castsi256_ps(a);

    return comp_return(g_test_mm256_castsi256_ps_data.expect, &dst, sizeof(dst));
}

int test_mm256_castsi256_si128()
{
    int32_t *src = g_test_mm256_castsi256_si128_data.a;
    __m256i a = _mm256_set_epi32(src[7], src[6], src[5], src[4], src[3], src[2], src[1], src[0]);
    __m128i dst = _mm256_castsi256_si128(a);

    return comp_return(g_test_mm256_castsi256_si128_data.expect, &dst, sizeof(dst));
}

int test_mm_cvtsi32_si128()
{
    int32_t a = g_test_mm_cvtsi32_si128_data.a;
    __m128i dst = _mm_cvtsi32_si128(a);

    return comp_return(g_test_mm_cvtsi32_si128_data.expect, &dst, sizeof(dst));
}

int test_mm_cvtsi128_si32()
{
    int32_t *src = g_test_mm_cvtsi128_si32_data.a;
    __m128i a = _mm_set_epi32(src[3], src[2], src[1], src[0]);
    int dst = _mm_cvtsi128_si32(a);

    return comp_return(&g_test_mm_cvtsi128_si32_data.expect, &dst, sizeof(dst));
}

int test_mm256_cvtepi32_pd()
{
    int32_t *src = g_test_mm256_cvtepi32_pd_data.a;

    __m128i a = _mm_set_epi32(src[3], src[2], src[1], src[0]);
    __m256d dst = _mm256_cvtepi32_pd(a);

    return comp_return(g_test_mm256_cvtepi32_pd_data.expect, &dst, sizeof(dst));
}

int test_mm256_cvtepi32_ps()
{
    int32_t *src = g_test_mm256_cvtepi32_ps_data.a;

    __m256i a = _mm256_set_epi32(src[7], src[6], src[5], src[4], src[3], src[2], src[1], src[0]);
    __m256 dst = _mm256_cvtepi32_ps(a);

    return comp_return(g_test_mm256_cvtepi32_ps_data.expect, &dst, sizeof(dst));
}

int test_mm_storeu_si128()
{
    int32_t *a = g_test_mm_storeu_si128_data.a;
    int32_t *expect = g_test_mm_storeu_si128_data.expect;
    __m128i ma, res;
    ma.vect_s32 = vld1q_s32(a);
    _mm_storeu_si128(&res, ma);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm256_store_si256()
{
    int32_t *a = g_test_mm256_store_si256_data.a;
    int32_t *expect = g_test_mm256_store_si256_data.expect;
    int iCount;
    __m256i ma, res;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    _mm256_store_si256(&res, ma);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_storeu_si256()
{
    int32_t *a = g_test_mm256_storeu_si256_data.a;
    int32_t *expect = g_test_mm256_storeu_si256_data.expect;
    int iCount;
    __m256i ma, res;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    _mm256_storeu_si256(&res, ma);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_stream_si256()
{
    int32_t *a = g_test_mm256_stream_si256_data.a;
    int32_t *expect = g_test_mm256_stream_si256_data.expect;
    int iCount;
    __m256i ma, res;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    _mm256_stream_si256(&res, ma);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_store_si512()
{
    int32_t *a = g_test_mm512_store_si512_data.a;
    int32_t *expect = g_test_mm512_store_si512_data.expect;
    int iCount;
    __m512i ma, res;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    _mm512_store_si512(&res, ma);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_storeu_si512()
{
    int32_t *a = g_test_mm512_storeu_si512_data.a;
    int32_t *expect = g_test_mm512_storeu_si512_data.expect;
    int iCount;
    __m512i ma, res;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    _mm512_storeu_si512(&res, ma);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_stream_si512()
{
    int32_t *a = g_test_mm512_stream_si512_data.a;
    int32_t *expect = g_test_mm512_stream_si512_data.expect;
    int iCount;
    __m512i ma, res;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    _mm512_stream_si512(&res, ma);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm256_inserti128_si256()
{
    int32_t *a = g_test_mm256_inserti128_si256_data.a;
    int32_t *b = g_test_mm256_inserti128_si256_data.b;
    int32_t *expect = g_test_mm256_inserti128_si256_data.expect;
    int iCount;
    __m256i ma, res;
    __m128i mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    mb.vect_s32 = vld1q_s32(b);
    res = _mm256_inserti128_si256(ma, mb, 0);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_insertf128_pd()
{
    float64_t *a = g_test_mm256_insertf128_pd_data.a;
    float64_t *b = g_test_mm256_insertf128_pd_data.b;
    int imm = g_test_mm256_insertf128_pd_data.imm;
    float64_t *expect = g_test_mm256_insertf128_pd_data.expect;
    int iCount;
    __m256d ma, res;
    __m128d mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
    }
    mb = vld1q_f64(b);
    res = _mm256_insertf128_pd(ma, mb, imm);

    return IsEqualFloat64x4(res, expect, DEFAULT_EPSILON_F64);
}
int test_mm256_insertf128_ps()
{
    float32_t *a = g_test_mm256_insertf128_ps_data.a;
    float32_t *b = g_test_mm256_insertf128_ps_data.b;
    int imm = g_test_mm256_insertf128_ps_data.imm;
    float32_t *expect = g_test_mm256_insertf128_ps_data.expect;
    int iCount;
    __m256 ma, res;
    __m128 mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
    }
    mb = vld1q_f32(b);
    res = _mm256_insertf128_ps(ma, mb, imm);

    return IsEqualFloat32x8(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm256_permute4x64_epi64()
{
    int64_t *a = g_test_mm256_permute4x64_epi64_data.a;
    int imm = g_test_mm256_permute4x64_epi64_data.imm;
    int64_t *expect = g_test_mm256_permute4x64_epi64_data.expect;
    int iCount;
    __m256i ma, res;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    res = _mm256_permute4x64_epi64(ma, imm);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_permute2f128_si256()
{
    int64_t *a = g_test_mm256_permute2f128_si256_data.a;
    int64_t *b = g_test_mm256_permute2f128_si256_data.b;
    int imm = g_test_mm256_permute2f128_si256_data.imm;
    int64_t *expect = g_test_mm256_permute2f128_si256_data.expect;
    int iCount;
    __m256i ma, mb, res;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
        mb.vect_s64[iCount] = vld1q_s64(b + iCount * 2);
    }
    res = _mm256_permute2f128_si256(ma, mb, imm);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_alignr_epi8()
{
    int8_t *a = g_test_mm256_alignr_epi8_data.a;
    int8_t *b = g_test_mm256_alignr_epi8_data.b;
    const int count = g_test_mm256_alignr_epi8_data.count;
    int8_t *expect = g_test_mm256_alignr_epi8_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s8[iCount] = vld1q_s8(a + iCount * 16);
        mb.vect_s8[iCount] = vld1q_s8(b + iCount * 16);
    }
    __m256i res = _mm256_alignr_epi8(ma, mb, count);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm_cmpestri()
{
    int16_t *a = g_test_mm_cmpestri_data_model_data.a;
    int16_t *b = g_test_mm_cmpestri_data_model_data.b;
    int la = g_test_mm_cmpestri_data_model_data.la;
    int lb = g_test_mm_cmpestri_data_model_data.lb;
    const int imm8 = g_test_mm_cmpestri_data_model_data.imm8;
    int expect = g_test_mm_cmpestri_data_model_data.expect;
    __m128i ma, mb;
    ma.vect_s16 = vld1q_s16(a);
    mb.vect_s16 = vld1q_s16(b);
    int res = _mm_cmpestri(ma, la, mb, lb, imm8);
    return res == expect;
}

int test_mm_cmpestrm()
{
    int16_t *a = g_test_mm_cmpestrm_data_model_data.a;
    int16_t *b = g_test_mm_cmpestrm_data_model_data.b;
    int la = g_test_mm_cmpestrm_data_model_data.la;
    int lb = g_test_mm_cmpestrm_data_model_data.lb;
    const int imm8 = g_test_mm_cmpestrm_data_model_data.imm8;
    int16_t *expect = g_test_mm_cmpestrm_data_model_data.expect;
    __m128i ma, mb, res;
    ma.vect_s16 = vld1q_s16(a);
    mb.vect_s16 = vld1q_s16(b);
    res = _mm_cmpestrm(ma, la, mb, lb, imm8);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_insert_epi32()
{
    __m128i a;
    int i = g_test_mm_insert_epi32_data.i;
    int32_t *expect = g_test_mm_insert_epi32_data.expect;

    a.vect_s32 = vld1q_s32(g_test_mm_insert_epi32_data.a);
    __m128i res = _mm_insert_epi32(a, i, 3);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm256_insert_epi32()
{
    __m256i a;
    int i = g_test_mm256_insert_epi32_data.i;
    int32_t *expect = g_test_mm256_insert_epi32_data.expect;

    for (unsigned int j = 0; j < M256_M128_NUM; j++) {
        a.vect_s32[j] = vld1q_s32(g_test_mm256_insert_epi32_data.a + j * M128I_INT32_NUM);
    }
    __m256i res = _mm256_insert_epi32(a, i, 6);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_insert_epi64()
{
    __m256i a;
    int64_t i = g_test_mm256_insert_epi64_data.i;
    int64_t *expect = g_test_mm256_insert_epi64_data.expect;

    for (unsigned int j = 0; j < M256_M128_NUM; j++) {
        a.vect_s64[j] = vld1q_s64(g_test_mm256_insert_epi64_data.a + j * M128I_INT64_NUM);
    }
    __m256i res = _mm256_insert_epi64(a, i, 3);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_castpd128_pd512()
{
    __m128d a = vld1q_f64(g_test_mm512_castpd128_pd512_data.a);
    __m512d res = _mm512_castpd128_pd512(a);

    return IsEqualFloat64x2(res.vect_f64[0], g_test_mm512_castpd128_pd512_data.expect, DEFAULT_EPSILON_F64);
}

int test_mm512_castpd512_pd128()
{
    __m512d a;
    double *expect = g_test_mm512_castpd512_pd128_data.expect;
    __m128d res;

    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        a.vect_f64[i] = vld1q_f64(g_test_mm512_castpd512_pd128_data.a + i * M128D_FLOAT64_NUM);
    }
    res = _mm512_castpd512_pd128(a);

    return IsEqualFloat64x2(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_castps128_ps512()
{
    __m128 a = vld1q_f32(g_test_mm512_castps128_ps512_data.a);
    __m512 res = _mm512_castps128_ps512(a);

    return IsEqualFloat32x4(res.vect_f32[0], g_test_mm512_castps128_ps512_data.expect, DEFAULT_EPSILON_F32);
}

int test_mm512_castps512_ps128()
{
    __m512 a;
    float *expect = g_test_mm512_castps512_ps128_data.expect;

    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        a.vect_f32[i] = vld1q_f32(g_test_mm512_castps512_ps128_data.a + i * M128_FLOAT32_NUM);
    }
    __m128 res = _mm512_castps512_ps128(a);

    return IsEqualFloat32x4(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_cvtepi32_pd()
{
    __m256i a;
    double *expect = g_test_mm512_cvtepi32_pd_data.expect;

    for (unsigned int i = 0; i < M256_M128_NUM; i++) {
        a.vect_s32[i] = vld1q_s32(g_test_mm512_cvtepi32_pd_data.a + i * M128I_INT32_NUM);
    }
    __m512d res = _mm512_cvtepi32_pd(a);
    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_cvtepi32_ps()
{
    __m512i a;
    float *expect = g_test_mm512_cvtepi32_ps_data.expect;

    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        a.vect_s32[i] = vld1q_s32(g_test_mm512_cvtepi32_ps_data.a + i * M128I_INT32_NUM);
    }
    __m512 res = _mm512_cvtepi32_ps(a);

    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_insertf32x8()
{
    __m512 a;
    __m256 b;
    int imm8 = g_test_mm512_insertf32x8_data.imm8;
    float *expect = g_test_mm512_insertf32x8_data.expect;

    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        a.vect_f32[i] = vld1q_f32(g_test_mm512_insertf32x8_data.a + i * M128_FLOAT32_NUM);
    }
    for (unsigned int i = 0; i < M256_M128_NUM; i++) {
        b.vect_f32[i] = vld1q_f32(g_test_mm512_insertf32x8_data.b + i * M128_FLOAT32_NUM);
    }
    __m512 res = _mm512_insertf32x8(a, b, imm8);
    return IsEqualFloat32x16(res, expect, DEFAULT_EPSILON_F32);
}

int test_mm512_insertf64x4()
{
    __m512d a;
    __m256d b;
    int imm8 = g_test_mm512_insertf64x4_data.imm8;
    double *expect = g_test_mm512_insertf64x4_data.expect;

    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        a.vect_f64[i] = vld1q_f64(g_test_mm512_insertf64x4_data.a + i * M128D_FLOAT64_NUM);
    }
    for (unsigned int i = 0; i < M256_M128_NUM; i++) {
        b.vect_f64[i] = vld1q_f64(g_test_mm512_insertf64x4_data.b + i * M128D_FLOAT64_NUM);
    }
    __m512d res = _mm512_insertf64x4(a, b, imm8);
    return IsEqualFloat64x8(res, expect, DEFAULT_EPSILON_F64);
}

int test_mm512_inserti32x8()
{
    __m512i a;
    __m256i b;
    int imm8 = g_test_mm512_inserti32x8_data.imm8;
    int32_t *expect = g_test_mm512_inserti32x8_data.expect;

    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        a.vect_s32[i] = vld1q_s32(g_test_mm512_inserti32x8_data.a + i * M128I_INT32_NUM);
    }
    for (unsigned int i = 0; i < M256_M128_NUM; i++) {
        b.vect_s32[i] = vld1q_s32(g_test_mm512_inserti32x8_data.b + i * M128I_INT32_NUM);
    }
    __m512i res = _mm512_inserti32x8(a, b, imm8);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_inserti64x4()
{
    __m512i a;
    __m256i b;
    int imm8 = g_test_mm512_inserti64x4_data.imm8;
    int64_t *expect = g_test_mm512_inserti64x4_data.expect;

    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        a.vect_s64[i] = vld1q_s64(g_test_mm512_inserti64x4_data.a + i * M128I_INT64_NUM);
    }
    for (unsigned int i = 0; i < M256_M128_NUM; i++) {
        b.vect_s64[i] = vld1q_s64(g_test_mm512_inserti64x4_data.b + i * M128I_INT64_NUM);
    }
    __m512i res = _mm512_inserti64x4(a, b, imm8);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm256_cmp_pd()
{
    __m256d source1, source2, dest;
    int i;
    long long expect[4];

    __m256d s1 = test_mm256_cmp_pd_data_model_unordered_data1;
    __m256d s2 = test_mm256_cmp_pd_data_model_unordered_data2;
    for (int j = 0; j < 32; j++) {
        MM256_CMP_PD(j, test_mm256_cmp_pd_data_model_unordered_ret[j][i], expect);
    }

    s1 = test_mm256_cmp_pd_data_model_ordered_data1;
    s2 = test_mm256_cmp_pd_data_model_ordered_data2;

    for (int j = 0; j < 32; j++) {
        MM256_CMP_PD(j, test_mm256_cmp_pd_data_model_ordered_ret[j][i], expect);
    }
    return TRUE;
}

int test_mm256_cmp_ps()
{
    __m256 source1, source2, dest;
    int i;
    int expect[8];

    __m256 s1 = test_mm256_cmp_ps_data_model_unordered_data1;
    __m256 s2 = test_mm256_cmp_ps_data_model_unordered_data2;
    for (int j = 0; j < 32; j++) {
        MM256_CMP_PS(j, test_mm256_cmp_ps_data_model_unordered_ret[j][i], expect);
    }

    s1 = test_mm256_cmp_ps_data_model_ordered_data1;
    s2 = test_mm256_cmp_ps_data_model_ordered_data2;

    for (int j = 0; j < 32; j++) {
        MM256_CMP_PS(j, test_mm256_cmp_ps_data_model_ordered_ret[j][i], expect);
    }
    return TRUE;
}

int test_mm512_cmp_pd_mask()
{
    __m512d a, b;
    __mmask8 result[32];

    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        a.vect_f64[i] = vld1q_f64(g_test_mm512_cmp_pd_mask_data1.a + i * M128D_FLOAT64_NUM);
        b.vect_f64[i] = vld1q_f64(g_test_mm512_cmp_pd_mask_data1.b + i * M128D_FLOAT64_NUM);
    }
    __mmask8* expect = g_test_mm512_cmp_pd_mask_data1.expect;
    for (int i = 0; i < 32; i++) {
        result[i] = _mm512_cmp_pd_mask(a, b, i);
    }
    if (!comp_return(result, expect, sizeof(__mmask8) * 32)) {
        return FALSE;
    }

    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        a.vect_f64[i] = vld1q_f64(g_test_mm512_cmp_pd_mask_data2.a + i * M128D_FLOAT64_NUM);
        b.vect_f64[i] = vld1q_f64(g_test_mm512_cmp_pd_mask_data2.b + i * M128D_FLOAT64_NUM);
    }
    expect = g_test_mm512_cmp_pd_mask_data2.expect;
    for (int i = 0; i < 32; i++) {
        result[i] = _mm512_cmp_pd_mask(a, b, i);
    }
    if (!comp_return(result, expect, sizeof(__mmask8) * 32)) {
        return FALSE;
    }
    return TRUE;
}

int test_mm512_cmp_ps_mask()
{
    __m512 a, b;
    __mmask16 result[32];

    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        a.vect_f32[i] = vld1q_f32(g_test_mm512_cmp_ps_mask_data1.a + i * M128_FLOAT32_NUM);
        b.vect_f32[i] = vld1q_f32(g_test_mm512_cmp_ps_mask_data1.b + i * M128_FLOAT32_NUM);
    }
    __mmask16* expect = g_test_mm512_cmp_ps_mask_data1.expect;
    for (int i = 0; i < 32; i++) {
        result[i] = _mm512_cmp_ps_mask(a, b, i);
    }
    if (!comp_return(result, expect, sizeof(__mmask16) * 32)) {
        return FALSE;
    }
    
    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        a.vect_f32[i] = vld1q_f32(g_test_mm512_cmp_ps_mask_data2.a + i * M128_FLOAT32_NUM);
        b.vect_f32[i] = vld1q_f32(g_test_mm512_cmp_ps_mask_data2.b + i * M128_FLOAT32_NUM);
    }
    expect = g_test_mm512_cmp_ps_mask_data2.expect;
    for (int i = 0; i < 32; i++) {
        result[i] = _mm512_cmp_ps_mask(a, b, i);
    }
    if (!comp_return(result, expect, sizeof(__mmask16) * 32)) {
        return FALSE;
    }
    return TRUE;
}

int test_mm512_permutexvar_epi32()
{
    __m512i idx;
    __m512i a;
    int32_t *expect = g_test_mm512_permutexvar_epi32_data.expect;

    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        idx.vect_s32[i] = vld1q_s32(g_test_mm512_permutexvar_epi32_data.idx + i * M128I_INT32_NUM);
    }
    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        a.vect_s32[i] = vld1q_s32(g_test_mm512_permutexvar_epi32_data.a + i * M128I_INT32_NUM);
    }
    __m512i res = _mm512_permutexvar_epi32(idx, a);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm_load_epi32()
{
    int32_t *a = g_test_mm_load_epi32_data.a;
    int32_t *expect = g_test_mm_load_epi32_data.expect;
    __m128i res = _mm_load_epi32((void const*)a);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_load_epi64()
{
    int64_t *a = g_test_mm_load_epi64_data.a;
    int64_t *expect = g_test_mm_load_epi64_data.expect;
    __m128i res = _mm_load_epi64((void const*)a);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_load_si128()
{
    int32_t *a = g_test_mm_load_si128_data.a;
    int32_t *expect = g_test_mm_load_si128_data.expect;
    __m128i res = _mm_load_si128((__m128i const*)a);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_load_ps()
{
    float32_t *a = g_test_mm_load_ps_data.a;
    float32_t *expect = g_test_mm_load_ps_data.expect;
    __m128 res = _mm_load_ps((float const*)a);
    return comp_return(expect, &res, sizeof(__m128));
}

int test_mm_load_pd()
{
    float64_t *a = g_test_mm_load_pd_data.a;
    float64_t *expect = g_test_mm_load_pd_data.expect;
    __m128d res = _mm_load_pd((double const*)a);
    return comp_return(expect, &res, sizeof(__m128d));
}

int test_mm256_load_epi32()
{
    int32_t *a = g_test_mm256_load_epi32_data.a;
    int32_t *expect = g_test_mm256_load_epi32_data.expect;
    __m256i res = _mm256_load_epi32((void const*)a);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_load_epi64()
{
    int64_t *a = g_test_mm256_load_epi64_data.a;
    int64_t *expect = g_test_mm256_load_epi64_data.expect;
    __m256i res = _mm256_load_epi64((void const*)a);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm256_load_ps()
{
    float32_t *a = g_test_mm256_load_ps_data.a;
    float32_t *expect = g_test_mm256_load_ps_data.expect;
    __m256 res = _mm256_load_ps((float const*)a);
    return comp_return(expect, &res, sizeof(__m256));
}

int test_mm256_load_pd()
{
    float64_t *a = g_test_mm256_load_pd_data.a;
    float64_t *expect = g_test_mm256_load_pd_data.expect;
    __m256d res = _mm256_load_pd((double const*)a);
    return comp_return(expect, &res, sizeof(__m256d));
}

int test_mm512_load_epi32()
{
    int32_t *a = g_test_mm512_load_epi32_data.a;
    int32_t *expect = g_test_mm512_load_epi32_data.expect;
    __m512i res = _mm512_load_epi32((void const*)a);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_load_epi64()
{
    int64_t *a = g_test_mm512_load_epi64_data.a;
    int64_t *expect = g_test_mm512_load_epi64_data.expect;
    __m512i res = _mm512_load_epi64((void const*)a);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm512_load_ps()
{
    float32_t *a = g_test_mm512_load_ps_data.a;
    float32_t *expect = g_test_mm512_load_ps_data.expect;
    __m512 res = _mm512_load_ps((float const*)a);
    return comp_return(expect, &res, sizeof(__m512));
}

int test_mm512_load_pd()
{
    float64_t *a = g_test_mm512_load_pd_data.a;
    float64_t *expect = g_test_mm512_load_pd_data.expect;
    __m512d res = _mm512_load_pd((double const*)a);
    return comp_return(expect, &res, sizeof(__m512d));
}

int test_mm_store_epi32()
{
    int32_t *a = g_test_mm_store_epi32_data.a;
    int32_t *expect = g_test_mm_store_epi32_data.expect;
    int32_t res[sizeof(__m128i) / sizeof(int32_t)];
    __m128i ma;
    ma.vect_s32 = vld1q_s32(a);
    _mm_store_epi32((void *)res, ma);
    return comp_return(expect, res, sizeof(__m128i));
}

int test_mm_store_epi64()
{
    int64_t *a = g_test_mm_store_epi64_data.a;
    int64_t *expect = g_test_mm_store_epi64_data.expect;
    int64_t res[sizeof(__m128i) / sizeof(int64_t)];
    __m128i ma;
    ma.vect_s64 = vld1q_s64(a);
    _mm_store_epi64((void *)res, ma);
    return comp_return(expect, res, sizeof(__m128i));
}

int test_mm_store_si128()
{
    int32_t *a = g_test_mm_store_si128_data.a;
    int32_t *expect = g_test_mm_store_si128_data.expect;
    __m128i ma, res;
    ma.vect_s32 = vld1q_s32(a);
    _mm_store_si128(&res, ma);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_store_pd()
{
    float64_t *a = g_test_mm_store_pd_data.a;
    float64_t *expect = g_test_mm_store_pd_data.expect;
    float64_t res[sizeof(__m128d) / sizeof(float64_t)];
    __m128d ma;
    ma = vld1q_f64(a);
    _mm_store_pd(res, ma);
    return comp_return(expect, res, sizeof(__m128d));
}

int test_mm_store_ps()
{
    float32_t *a = g_test_mm_store_ps_data.a;
    float32_t *expect = g_test_mm_store_ps_data.expect;
    float32_t res[sizeof(__m128) / sizeof(float32_t)];
    __m128 ma;
    ma = vld1q_f32(a);
    _mm_store_ps(res, ma);
    return comp_return(expect, res, sizeof(__m128));
}

int test_mm256_store_epi32()
{
    int32_t *a = g_test_mm256_store_epi32_data.a;
    int32_t *expect = g_test_mm256_store_epi32_data.expect;
    int iCount;
    int32_t res[sizeof(__m256i) / sizeof(int32_t)];
    __m256i ma;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    _mm256_store_epi32((void *)res, ma);
    return comp_return(expect, res, sizeof(__m256i));
}

int test_mm256_store_epi64()
{
    int64_t *a = g_test_mm256_store_epi64_data.a;
    int64_t *expect = g_test_mm256_store_epi64_data.expect;
    int iCount;
    int64_t res[sizeof(__m256i) / sizeof(int64_t)];
    __m256i ma;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    _mm256_store_epi64((void *)res, ma);
    return comp_return(expect, res, sizeof(__m256i));
}

int test_mm256_store_pd()
{
    float64_t *a = g_test_mm256_store_pd_data.a;
    float64_t *expect = g_test_mm256_store_pd_data.expect;
    int iCount;
    float64_t res[sizeof(__m256d) / sizeof(float64_t)];
    __m256d ma;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
    }
    _mm256_store_pd(res, ma);
    return comp_return(expect, res, sizeof(__m256d));
}

int test_mm256_store_ps()
{
    float32_t *a = g_test_mm256_store_ps_data.a;
    float32_t *expect = g_test_mm256_store_ps_data.expect;
    int iCount;
    float32_t res[sizeof(__m256) / sizeof(float32_t)];
    __m256 ma;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
    }
    _mm256_store_ps(res, ma);
    return comp_return(expect, res, sizeof(__m256));
}

int test_mm512_store_epi32()
{
    int32_t *a = g_test_mm512_store_epi32_data.a;
    int32_t *expect = g_test_mm512_store_epi32_data.expect;
    int iCount;
    int32_t res[sizeof(__m512i) / sizeof(int32_t)];
    __m512i ma;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
    }
    _mm512_store_epi32((void *)res, ma);
    return comp_return(expect, res, sizeof(__m512i));
}

int test_mm512_store_epi64()
{
    int64_t *a = g_test_mm512_store_epi64_data.a;
    int64_t *expect = g_test_mm512_store_epi64_data.expect;
    int iCount;
    int64_t res[sizeof(__m512i) / sizeof(int64_t)];
    __m512i ma;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s64[iCount] = vld1q_s64(a + iCount * 2);
    }
    _mm512_store_epi64((void *)res, ma);
    return comp_return(expect, res, sizeof(__m512i));
}

int test_mm512_store_pd()
{
    float64_t *a = g_test_mm512_store_pd_data.a;
    float64_t *expect = g_test_mm512_store_pd_data.expect;
    int iCount;
    float64_t res[sizeof(__m512d) / sizeof(float64_t)];
    __m512d ma;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f64[iCount] = vld1q_f64(a + iCount * 2);
    }
    _mm512_store_pd(res, ma);
    return comp_return(expect, res, sizeof(__m512d));
}

int test_mm512_store_ps()
{
    float32_t *a = g_test_mm512_store_ps_data.a;
    float32_t *expect = g_test_mm512_store_ps_data.expect;
    int iCount;
    float32_t res[sizeof(__m512) / sizeof(float32_t)];
    __m512 ma;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_f32[iCount] = vld1q_f32(a + iCount * 4);
    }
    _mm512_store_ps(res, ma);
    return comp_return(expect, res, sizeof(__m512));
}

int test_mm512_permutex2var_epi32()
{
    __m512i idx;
    __m512i a, b;
    int32_t *expect = g_test_mm512_permutex2var_epi32_data.expect;

    for (unsigned int i = 0; i < M512_M128_NUM; i++) {
        idx.vect_s32[i] = vld1q_s32(g_test_mm512_permutex2var_epi32_data.idx + i * M128I_INT32_NUM);
        a.vect_s32[i] = vld1q_s32(g_test_mm512_permutex2var_epi32_data.a + i * M128I_INT32_NUM);
        b.vect_s32[i] = vld1q_s32(g_test_mm512_permutex2var_epi32_data.b + i * M128I_INT32_NUM);
    }

    __m512i res = _mm512_permutex2var_epi32(a, idx, b);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm_max_epu8()
{
    uint8_t *a = g_test_mm_max_epu8_data.a;
    uint8_t *b = g_test_mm_max_epu8_data.b;
    uint8_t *expect = g_test_mm_max_epu8_data.expect;
    __m128i ma, mb;
    ma.vect_u8 = vld1q_u8(a);
    mb.vect_u8 = vld1q_u8(b);
    __m128i res = _mm_max_epu8(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_min_epu8()
{
    uint8_t *a = g_test_mm_min_epu8_data.a;
    uint8_t *b = g_test_mm_min_epu8_data.b;
    uint8_t *expect = g_test_mm_min_epu8_data.expect;
    __m128i ma, mb;
    ma.vect_u8 = vld1q_u8(a);
    mb.vect_u8 = vld1q_u8(b);
    __m128i res = _mm_min_epu8(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm256_max_epi32()
{
    int32_t *a = g_test_mm256_max_epi32_data.a;
    int32_t *b = g_test_mm256_max_epi32_data.b;
    int32_t *expect = g_test_mm256_max_epi32_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_max_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_max_epi32()
{
    int32_t *a = g_test_mm512_max_epi32_data.a;
    int32_t *b = g_test_mm512_max_epi32_data.b;
    int32_t *expect = g_test_mm512_max_epi32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_max_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm_packs_epi16()
{
    int16_t *a = g_test_mm_packs_epi16_data.a;
    int16_t *b = g_test_mm_packs_epi16_data.b;
    int16_t *expect = g_test_mm_packs_epi16_data.expect;
    __m128i ma, mb;
    ma.vect_s16 = vld1q_s16(a);
    mb.vect_s16 = vld1q_s16(b);
    __m128i res = _mm_packs_epi16(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm_packs_epi32()
{
    int32_t *a = g_test_mm_packs_epi32_data.a;
    int32_t *b = g_test_mm_packs_epi32_data.b;
    int32_t *expect = g_test_mm_packs_epi32_data.expect;
    __m128i ma, mb;
    ma.vect_s32 = vld1q_s32(a);
    mb.vect_s32 = vld1q_s32(b);
    __m128i res = _mm_packs_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m128i));
}

int test_mm256_packs_epi32()
{
    int32_t *a = g_test_mm256_packs_epi32_data.a;
    int32_t *b = g_test_mm256_packs_epi32_data.b;
    int32_t *expect = g_test_mm256_packs_epi32_data.expect;
    int iCount;
    __m256i ma, mb;
    for (iCount = 0; iCount < g_256bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m256i res = _mm256_packs_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m256i));
}

int test_mm512_packs_epi32()
{
    int32_t *a = g_test_mm512_packs_epi32_data.a;
    int32_t *b = g_test_mm512_packs_epi32_data.b;
    int32_t *expect = g_test_mm512_packs_epi32_data.expect;
    int iCount;
    __m512i ma, mb;
    for (iCount = 0; iCount < g_512bit_divto_128bit; iCount++) {
        ma.vect_s32[iCount] = vld1q_s32(a + iCount * 4);
        mb.vect_s32[iCount] = vld1q_s32(b + iCount * 4);
    }
    __m512i res = _mm512_packs_epi32(ma, mb);
    return comp_return(expect, &res, sizeof(__m512i));
}

int test_mm_malloc()
{
    size_t size = g_test_mm_malloc_data.size;
    size_t align = g_test_mm_malloc_data.align;

    void *p = _mm_malloc(size, align);
    if (p == NULL)
        return FALSE;
    int res = (uintptr_t)p % align == 0 ? TRUE : FALSE;
    _mm_free(p);
    return res;
}
