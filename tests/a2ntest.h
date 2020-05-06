/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
 * Description: avx2neon test head file
 * Author: xuqimeng
 * Create: 2019-11-05
*/
#ifndef AVX2NEON_TEST_H
#define AVX2NEON_TEST_H

#define TRUE 1
#define FALSE 0

#define DEFAULT_EPSILON_F32 1e-4
#define DEFAULT_EPSILON_F64 1e-8

#include "avx2neon.h"

typedef enum {
    UT_MM_EXTRACT_EPI32,
    UT_MM_EXTRACT_EPI64,
    UT_MM_MOVE_SD,
    UT_MM_MOVE_SS,
    UT_MM256_ADD_EPI16,
    UT_MM256_ADD_EPI32,
    UT_MM256_ADD_EPI64,
    UT_MM256_ADD_EPI8,
    UT_MM256_ADD_PD,
    UT_MM256_ADD_PS,
    UT_MM_ADDS_EPU8,
    UT_MM256_ADDS_EPI16,
    UT_MM256_ADDS_EPI8,
    UT_MM256_ADDS_EPU16,
    UT_MM256_ADDS_EPU8,
    UT_MM256_ADDSUB_PD,
    UT_MM256_ADDSUB_PS,
    UT_MM256_BLENDV_PD,
    UT_MM256_BLENDV_PS,
    UT_MM256_BLEND_PD,
    UT_MM256_BLEND_PS,
    UT_MM512_MASK_BLEND_EPI32,
    UT_MM512_MASK_BLEND_PD,
    UT_MM512_MASK_BLEND_PS,
    UT_MM256_CASTPD128_PD256,
    UT_MM256_CASTPD256_PD128,
    UT_MM256_CASTPS128_PS256,
    UT_MM256_CASTPS256_PS128,
    UT_MM_CVTSI32_SI128,
    UT_MM_CVTSI128_SI32,
    UT_MM256_CVTEPI32_PD,
    UT_MM256_CVTEPI32_PS,
    UT_MM256_DIV_EPI16,
    UT_MM256_DIV_EPI32,
    UT_MM256_DIV_EPI64,
    UT_MM256_DIV_EPI8,
    UT_MM256_DIV_EPU16,
    UT_MM256_DIV_EPU32,
    UT_MM256_DIV_EPU64,
    UT_MM256_DIV_EPU8,
    UT_MM256_DIV_PD,
    UT_MM256_DIV_PS,
    UT_MM256_EXTRACT_EPI32,
    UT_MM256_EXTRACT_EPI64,
    UT_MM256_EXTRACTF128_PD,
    UT_MM256_EXTRACTF128_PS,
    UT_MM256_INSERTF128_PD,
    UT_MM256_INSERTF128_PS,
    UT_MM256_MUL_EPI32,
    UT_MM256_MUL_EPU32,
    UT_MM256_MUL_PD,
    UT_MM256_MUL_PS,
    UT_MM256_MULHI_EPI16,
    UT_MM256_MULHI_EPU16,
    UT_MM256_MULHRS_EPI16,
    UT_MM256_MULLO_EPI16,
    UT_MM256_MULLO_EPI32,
    UT_MM256_MULLO_EPI64,
    UT_MM256_MULTISHIFT_EPI64_EPI8,
    UT_MM256_SET1_EPI32,
    UT_MM256_SET1_PD,
    UT_MM256_SET1_PS,
    UT_MM_SUB_EPI8,
    UT_MM256_SUB_EPI16,
    UT_MM256_SUB_EPI32,
    UT_MM256_SUB_EPI64,
    UT_MM256_SUB_EPI8,
    UT_MM256_SUB_PD,
    UT_MM256_SUB_PS,
    UT_MM256_SUBS_EPI16,
    UT_MM256_SUBS_EPI8,
    UT_MM256_SUBS_EPU16,
    UT_MM256_SUBS_EPU8,
    UT_MM512_ADD_EPI16,
    UT_MM512_ADD_EPI32,
    UT_MM512_ADD_EPI64,
    UT_MM512_ADD_EPI8,
    UT_MM512_ADD_PD,
    UT_MM512_ADD_PS,
    UT_MM512_ADD_ROUND_PD,
    UT_MM512_ADD_ROUND_PS,
    UT_MM512_ADDN_PD,
    UT_MM512_ADDN_PS,
    UT_MM512_ADDN_ROUND_PD,
    UT_MM512_ADDN_ROUND_PS,
    UT_MM512_ADDS_EPI16,
    UT_MM512_ADDS_EPI8,
    UT_MM512_ADDS_EPU16,
    UT_MM512_ADDS_EPU8,
    UT_MM512_ADDSETC_EPI32,
    UT_MM512_ADDSETS_EPI32,
    UT_MM512_ADDSETS_PS,
    UT_MM512_ADDSETS_ROUND_PS,
    UT_MM512_DIV_EPI16,
    UT_MM512_DIV_EPI32,
    UT_MM512_DIV_EPI64,
    UT_MM512_DIV_EPI8,
    UT_MM512_DIV_EPU16,
    UT_MM512_DIV_EPU32,
    UT_MM512_DIV_EPU64,
    UT_MM512_DIV_EPU8,
    UT_MM512_DIV_PD,
    UT_MM512_DIV_PS,
    UT_MM512_DIV_ROUND_PD,
    UT_MM512_DIV_ROUND_PS,
    UT_MM512_EXTRACTF32x8_PS,
    UT_MM512_EXTRACTF64x4_PD,
    UT_MM512_MUL_EPI32,
    UT_MM512_MUL_EPU32,
    UT_MM512_MUL_PD,
    UT_MM512_MUL_PS,
    UT_MM512_MUL_ROUND_PD,
    UT_MM512_MUL_ROUND_PS,
    UT_MM512_MULHI_EPI16,
    UT_MM512_MULHI_EPI32,
    UT_MM512_MULHI_EPU16,
    UT_MM512_MULHI_EPU32,
    UT_MM512_MULHRS_EPI16,
    UT_MM512_MULLO_EPI16,
    UT_MM512_MULLO_EPI32,
    UT_MM512_MULLO_EPI64,
    UT_MM512_MULLOX_EPI64,
    UT_MM512_MULTISHIFT_EPI64_EPI8,
    UT_MM512_SUB_EPI16,
    UT_MM512_SUB_EPI32,
    UT_MM512_SUB_EPI64,
    UT_MM512_SUB_EPI8,
    UT_MM512_SUB_PD,
    UT_MM512_SUB_PS,
    UT_MM512_SUB_ROUND_PD,
    UT_MM512_SUB_ROUND_PS,
    UT_MM512_SUBR_EPI32,
    UT_MM512_SUBR_PD,
    UT_MM512_SUBR_PS,
    UT_MM512_SUBR_ROUND_PD,
    UT_MM512_SUBR_ROUND_PS,
    UT_MM512_SUBRSETB_EPI32,
    UT_MM512_SUBS_EPI16,
    UT_MM512_SUBS_EPI8,
    UT_MM512_SUBS_EPU16,
    UT_MM512_SUBS_EPU8,
    UT_MM512_SUBSETB_EPI32,
    UT_MM256_ALIGNR_EPI8,
    UT_MM_AND_SI128,
    UT_MM256_AND_SI256,
    UT_MM_ANDNOT_SI128,
    UT_MM256_ANDNOT_SI256,
    UT_MM256_BROADCASTQ_EPI64,
    UT_MM256_BROADCASTSI128_SI256,
    UT_MM_CASTSI128_PS,
    UT_MM256_CASTSI128_SI256,
    UT_MM256_CASTSI256_PS,
    UT_MM256_CASTSI256_SI128,
    UT_MM_CMPEQ_EPI8,
    UT_MM_CMPEQ_EPI32,
    UT_MM256_CMPGT_EPI32,
    UT_MM256_CMPEQ_EPI32,
    UT_MM256_CMPEQ_EPI8,
    UT_MM256_EXTRACTI128_SI256,
    UT_MM256_INSERTI128_SI256,
    UT_MM_LOADU_SI128,
    UT_MM256_LOAD_SI256,
    UT_MM256_LOADU_SI256,
    UT_MM256_MASKLOAD_EPI32,
    UT_MM_MOVEMASK_EPI8,
    UT_MM_MOVEMASK_PS,
    UT_MM256_MOVEMASK_EPI8,
    UT_MM256_MOVEMASK_PS,
    UT_MM_OR_SI128,
    UT_MM256_OR_SI256,
    UT_MM256_OR_PS,
    UT_MM256_OR_PD,
    UT_MM256_PERMUTE4X64_EPI64,
    UT_MM256_PERMUTE2F128_SI256,
    UT_MM_SET1_EPI8,
    UT_MM_SET1_EPI32,
    UT_MM_SET1_PS,
    UT_MM256_SET_EPI64X,
    UT_MM256_SET_M128I,
    UT_MM256_SET1_EPI64X,
    UT_MM256_SET1_EPI8,
    UT_MM_SETZERO_SI128,
    UT_MM256_SETZERO_SI256,
    UT_MM512_SETZERO_SI512,
    UT_MM256_SET_PS,
    UT_MM256_SET_PD,
    UT_MM256_SETZERO_PS,
    UT_MM256_SETZERO_PD,
    UT_MM_SHUFFLE_EPI8,
    UT_MM256_SHUFFLE_EPI8,
    UT_MM_SLL_EPI64,
    UT_MM_SLLI_SI128,
    UT_MM_SRLI_SI128,
    UT_MM_SLLI_EPI32,
    UT_MM_SLLI_EPI64,
    UT_MM_SRLI_EPI64,
    UT_MM256_SLL_EPI32,
    UT_MM256_SLL_EPI64,
    UT_MM256_SLLI_EPI64,
    UT_MM256_SLLI_SI256,
    UT_MM256_SLLI_EPI32,
    UT_MM256_SRLI_EPI64,
    UT_MM256_SRLI_SI256,
    UT_MM_STOREU_SI128,
    UT_MM256_STORE_SI256,
    UT_MM256_STOREU_SI256,
    UT_MM256_STREAM_SI256,
    UT_MM256_TESTZ_SI256,
    UT_MM256_UNPACKHI_EPI8,
    UT_MM256_UNPACKLO_EPI8,
    UT_MM_XOR_SI128,
    UT_MM256_XOR_SI256,
    UT_MM256_ZEROUPPER,
    UT_MM512_ABS_EPI8,
    UT_MM512_AND_SI512,
    UT_MM512_ANDNOT_SI512,
    UT_MM512_BROADCAST_I32X4,
    UT_MM512_BROADCAST_I64X4,
    UT_MM512_BSLLI_EPI128,
    UT_MM512_BSRLI_EPI128,
    UT_MM512_CMP_EPI32_MASK,
    UT_MM512_CMP_EPI8_MASK,
    UT_MM512_CMPEQ_EPI8_MASK,
    UT_MM512_CMPGT_EPI32_MASK,
    UT_MM512_EXTRACTI32X4_EPI32,
    UT_MM512_LOAD_SI512,
    UT_MM512_LOADU_SI512,
    UT_MM512_MASK_BROADCAST_I64X4,
    UT_MM512_MASK_CMPEQ_EPI8_MASK,
    UT_MM512_MASK_LOADU_EPI8,
    UT_MM512_MASKZ_LOADU_EPI8,
    UT_MM512_MASKZ_SHUFFLE_EPI8,
    UT_MM512_MOVM_EPI8,
    UT_MM512_MOVM_EPI32,
    UT_MM512_OR_SI512,
    UT_MM512_PERMUTEXVAR_EPI64,
    UT_MM512_SET_EPI32,
    UT_MM512_SET_EPI64,
    UT_MM512_SET1_EPI32,
    UT_MM512_SET1_EPI64,
    UT_MM512_SET1_EPI8,
    UT_MM512_SET_PS,
    UT_MM512_SET_PD,
    UT_MM512_SET1_PS,
    UT_MM512_SET1_PD,
    UT_MM512_SETZERO_PS,
    UT_MM512_SETZERO_PD,
    UT_MM512_SHUFFLE_EPI8,
    UT_MM512_SLL_EPI64,
    UT_MM512_SLLI_EPI64,
    UT_MM512_SRLI_EPI64,
    UT_MM512_STORE_SI512,
    UT_MM512_STOREU_SI512,
    UT_MM512_STREAM_SI512,
    UT_MM512_TEST_EPI8_MASK,
    UT_MM512_TEST_EPI32_MASK,
    UT_MM512_TEST_EPI64_MASK,
    UT_MM512_UNPACKHI_EPI8,
    UT_MM512_UNPACKLO_EPI8,
    UT_MM512_XOR_SI512,
    UT_MM512_AND_EPI32,
    UT_MM512_AND_EPI64,
    UT_MM512_OR_EPI32,
    UT_MM512_OR_EPI64,
    UT_MM512_XOR_PS,
    UT_MM512_XOR_PD,
    UT_MM_CMPEQ_EPI64,
    UT_MM_CMPESTRI,
    UT_MM_CMPESTRM,
    UT_MM_CRC32_U16,
    UT_MM_CRC32_U32,
    UT_MM_CRC32_U64,
    UT_MM_CRC32_U8,
    UT_MM_EXTRACT_PS,
    UT_MM_POPCNT_U32,
    UT_MM_POPCNT_U64,
    UT_MM_SET_PD,
    UT_MM_SET1_EPI64X,
    UT_MM_SET1_PD,
    UT_MM_TESTZ_SI128,
    UT_MM256_CMP_PD,
    UT_MM256_CMP_PS,
    UT_MM512_CMP_PD_MASK,
    UT_MM512_CMP_PS_MASK,
    UT_MM_INSERT_EPI32,
    UT_MM256_INSERT_EPI32,
    UT_MM256_INSERT_EPI64,
    UT_MM512_CASTPD128_PD512,
    UT_MM512_CASTPD512_PD128,
    UT_MM512_CASTPS128_PS512,
    UT_MM512_CASTPS512_PS128,
    UT_MM512_CVTEPI32_PD,
    UT_MM512_CVTEPI32_PS,
    UT_MM512_INSERTF32X8,
    UT_MM512_INSERTF64X4,
    UT_MM512_INSERTI32X8,
    UT_MM512_INSERTI64X4,
    UT_MM512_PERMUTEXVAR_EPI32,
    UT_MM512_PERMUTEX2VAR_EPI32,
	UT_MM_LOAD_EPI32,
    UT_MM_LOAD_EPI64,
    UT_MM_LOAD_SI128,
    UT_MM_LOAD_PD,
    UT_MM_LOAD_PS,
    UT_MM256_LOAD_EPI32,
    UT_MM256_LOAD_EPI64,
    UT_MM256_LOAD_PD,
    UT_MM256_LOAD_PS,
    UT_MM512_LOAD_EPI32,
    UT_MM512_LOAD_EPI64,
    UT_MM512_LOAD_PD,
    UT_MM512_LOAD_PS,
    UT_MM_STORE_EPI32,
    UT_MM_STORE_EPI64,
    UT_MM_STORE_SI128,
    UT_MM_STORE_PD,
    UT_MM_STORE_PS,
    UT_MM256_STORE_EPI32,
    UT_MM256_STORE_EPI64,
    UT_MM256_STORE_PD,
    UT_MM256_STORE_PS,
    UT_MM512_STORE_EPI32,
    UT_MM512_STORE_EPI64,
    UT_MM512_STORE_PD,
    UT_MM512_STORE_PS,
    UT_MM256_SET_EPI32,
    UT_MM_MAX_EPU8,
    UT_MM_MIN_EPU8,
    UT_MM256_MAX_EPI32,
    UT_MM512_MAX_EPI32,
    UT_MM_PACKS_EPI16,
    UT_MM_PACKS_EPI32,
    UT_MM256_PACKS_EPI32,
    UT_MM512_PACKS_EPI32,
    UT_MM_MALLOC
} InstructionTest;

const char *RunTest(InstructionTest test, int *flag);

int IsEqualFloat32x4(__m128 a, const float32_t *x, float epsilon);
int IsEqualFloat64x2(__m128d a, const float64_t *x, float epsilon);

int test_mm_popcnt_u32();
int test_mm_popcnt_u64();
int test_mm256_div_epi8();
int test_mm256_div_epi16();
int test_mm256_div_epi32();
int test_mm256_div_epi64();
int test_mm256_div_epu8();
int test_mm256_div_epu16();
int test_mm256_div_epu32();
int test_mm256_div_epu64();
int test_mm256_div_ps();
int test_mm256_div_pd();
int test_mm512_div_ps();
int test_mm512_div_pd();
int test_mm256_add_epi8();
int test_mm256_add_epi16();
int test_mm256_add_epi32();
int test_mm256_add_epi64();
int test_mm512_add_epi8();
int test_mm512_add_epi16();
int test_mm512_add_epi32();
int test_mm512_add_epi64();
int test_mm_adds_epu8();
int test_mm256_adds_epi8();
int test_mm256_adds_epi16();
int test_mm256_adds_epu8();
int test_mm256_adds_epu16();
int test_mm512_adds_epi8();
int test_mm512_adds_epi16();
int test_mm512_adds_epu8();
int test_mm512_adds_epu16();
int test_mm256_add_ps();
int test_mm256_add_pd();
int test_mm512_add_ps();
int test_mm512_add_pd();
int test_mm512_add_round_ps();
int test_mm512_add_round_pd();
int test_mm512_addn_ps();
int test_mm512_addn_pd();
int test_mm512_addn_round_ps();
int test_mm512_addn_round_pd();
int test_mm512_addsetc_epi32();
int test_mm512_addsets_epi32();
int test_mm512_addsets_ps();
int test_mm512_addsets_round_ps();
int test_mm256_addsub_ps();
int test_mm256_addsub_pd();
int test_mm256_blendv_ps();
int test_mm256_blendv_pd();
int test_mm256_blend_ps();
int test_mm256_blend_pd();
int test_mm512_mask_blend_epi32();
int test_mm512_mask_blend_ps();
int test_mm512_mask_blend_pd();
int test_mm_sub_epi8();
int test_mm256_sub_epi16();
int test_mm256_sub_epi32();
int test_mm256_sub_epi64();
int test_mm256_sub_epi8();
int test_mm256_sub_pd();
int test_mm256_sub_ps();
int test_mm512_sub_epi16();
int test_mm512_sub_epi32();
int test_mm512_sub_epi64();
int test_mm512_sub_epi8();
int test_mm512_sub_pd();
int test_mm512_sub_ps();
int test_mm256_subs_epi16();
int test_mm256_subs_epi8();
int test_mm256_subs_epu16();
int test_mm256_subs_epu8();
int test_mm512_subs_epi16();
int test_mm512_subs_epi8();
int test_mm512_subs_epu16();
int test_mm512_subs_epu8();
int test_mm512_sub_round_pd();
int test_mm512_sub_round_ps();
int test_mm512_subr_epi32();
int test_mm512_subr_ps();
int test_mm512_subr_pd();
int test_mm512_subr_round_ps();
int test_mm512_subr_round_pd();
int test_mm512_subsetb_epi32();
int test_mm512_subrsetb_epi32();
int test_mm256_zeroupper();
int test_mm512_bslli_epi128();
int test_mm512_bsrli_epi128();
int test_mm512_permutexvar_epi64();
int test_mm512_extracti32x4_epi32();
int test_mm512_test_epi8_mask();
int test_mm512_test_epi32_mask();
int test_mm512_test_epi64_mask();
int test_mm256_mul_epi32();
int test_mm256_mul_epu32();
int test_mm256_mul_pd();
int test_mm256_mul_ps();
int test_mm256_mulhi_epi16();
int test_mm256_mulhi_epu16();
int test_mm512_mul_epi32();
int test_mm512_mul_epu32();
int test_mm512_mul_pd();
int test_mm512_mul_ps();
int test_mm512_mulhi_epi16();
int test_mm512_mulhi_epu16();
int test_mm512_mulhi_epi32();
int test_mm512_mulhi_epu32();
int test_mm256_mullo_epi16();
int test_mm256_mullo_epi32();
int test_mm256_mullo_epi64();
int test_mm512_mullo_epi16();
int test_mm512_mullo_epi32();
int test_mm512_mullo_epi64();
int test_mm512_mullox_epi64();
int test_mm256_mulhrs_epi16();
int test_mm512_mulhrs_epi16();
int test_mm512_mul_round_pd();
int test_mm512_mul_round_ps();
int test_mm_sll_epi64();
int test_mm_slli_si128();
int test_mm_srli_si128();
int test_mm_slli_epi32();
int test_mm_slli_epi64();
int test_mm_srli_epi64();
int test_mm256_sll_epi32();
int test_mm256_sll_epi64();
int test_mm512_sll_epi64();
int test_mm256_slli_epi32();
int test_mm256_slli_epi64();
int test_mm512_slli_epi64();
int test_mm256_srli_epi64();
int test_mm512_srli_epi64();
int test_mm256_slli_si256();
int test_mm256_srli_si256();
int test_mm_and_si128();
int test_mm256_and_si256();
int test_mm512_and_si512();
int test_mm_or_si128();
int test_mm256_or_si256();
int test_mm512_or_si512();
int test_mm_andnot_si128();
int test_mm256_andnot_si256();
int test_mm512_andnot_si512();
int test_mm_xor_si128();
int test_mm256_xor_si256();
int test_mm512_xor_si512();
int test_mm256_or_ps();
int test_mm256_or_pd();
int test_mm512_and_epi32();
int test_mm512_and_epi64();
int test_mm512_or_epi32();
int test_mm512_or_epi64();
int test_mm512_xor_ps();
int test_mm512_xor_pd();
int test_mm256_cmpgt_epi32();
int test_mm256_cmpeq_epi8();
int test_mm256_cmpeq_epi32();
int test_mm_cmpeq_epi8();
int test_mm_cmpeq_epi32();
int test_mm_cmpeq_epi64();
int test_mm512_set_epi32();
int test_mm512_set_epi64();
int test_mm512_set1_epi32();
int test_mm512_set1_epi64();
int test_mm512_set1_epi8();
int test_mm512_set_ps();
int test_mm512_set_pd();
int test_mm512_set1_ps();
int test_mm512_set1_pd();
int test_mm512_setzero_ps();
int test_mm512_setzero_pd();
int test_mm_move_sd();
int test_mm_move_ss();
int test_mm_movemask_epi8();
int test_mm_movemask_ps();
int test_mm256_movemask_epi8();
int test_mm256_movemask_ps();
int test_mm_testz_si128();
int test_mm256_testz_si256();
int test_mm512_movm_epi8();
int test_mm512_movm_epi32();
int test_mm_extract_epi32();
int test_mm_extract_epi64();
int test_mm256_extracti128_si256();
int test_mm_extract_ps();
int test_mm256_extract_epi32();
int test_mm256_extract_epi64();
int test_mm256_extractf128_ps();
int test_mm256_extractf128_pd();
int test_mm512_extractf32x8_ps();
int test_mm512_extractf64x4_pd();
int test_mm_crc32_u8();
int test_mm_crc32_u16();
int test_mm_crc32_u32();
int test_mm_crc32_u64();
int test_mm_shuffle_epi8();
int test_mm256_shuffle_epi8();
int test_mm512_shuffle_epi8();
int test_mm512_maskz_shuffle_epi8();
int test_mm256_multishift_epi64_epi8();
int test_mm512_multishift_epi64_epi8();
int test_mm512_cmp_epi32_mask();
int test_mm512_cmp_epi8_mask();
int test_mm512_cmpeq_epi8_mask();
int test_mm512_cmpgt_epi32_mask();
int test_mm512_mask_cmpeq_epi8_mask();
int test_mm256_unpacklo_epi8();
int test_mm256_unpackhi_epi8();
int test_mm512_unpacklo_epi8();
int test_mm512_unpackhi_epi8();
int test_mm_storeu_si128();
int test_mm256_store_si256();
int test_mm256_storeu_si256();
int test_mm256_stream_si256();
int test_mm512_store_si512();
int test_mm512_storeu_si512();
int test_mm512_stream_si512();
int test_mm256_inserti128_si256();
int test_mm256_insertf128_pd();
int test_mm256_insertf128_ps();
int test_mm256_permute4x64_epi64();
int test_mm256_permute2f128_si256();
int test_mm_set_pd(void);
int test_mm256_set_epi32(void);
int test_mm256_set_epi64x(void);
int test_mm256_set_m128i(void);
int test_mm256_set_ps(void);
int test_mm256_set_pd(void);
int test_mm_setzero_si128(void);
int test_mm256_setzero_si256(void);
int test_mm512_setzero_si512(void);
int test_mm256_setzero_ps(void);
int test_mm256_setzero_pd(void);
int test_mm_set1_epi8(void);
int test_mm_set1_epi32(void);
int test_mm_set1_ps(void);
int test_mm_set1_epi64x(void);
int test_mm_set1_pd(void);
int test_mm256_set1_epi8(void);
int test_mm256_set1_epi32(void);
int test_mm256_set1_epi64x(void);
int test_mm256_set1_pd(void);
int test_mm256_set1_ps(void);
int test_mm256_alignr_epi8(void);
int test_mm_loadu_si128(void);
int test_mm256_load_si256(void);
int test_mm256_loadu_si256(void);
int test_mm256_maskload_epi32(void);
int test_mm512_load_si512(void);
int test_mm512_loadu_si512(void);
int test_mm512_mask_loadu_epi8(void);
int test_mm512_maskz_loadu_epi8(void);
int test_mm512_abs_epi8(void);
int test_mm256_broadcastq_epi64(void);
int test_mm256_broadcastsi128_si256(void);
int test_mm512_broadcast_i32x4(void);
int test_mm512_broadcast_i64x4(void);
int test_mm512_mask_broadcast_i64x4(void);
int test_mm256_castpd128_pd256(void);
int test_mm256_castpd256_pd128(void);
int test_mm256_castps128_ps256(void);
int test_mm256_castps256_ps128(void);
int test_mm_castsi128_ps(void);
int test_mm256_castsi128_si256(void);
int test_mm256_castsi256_ps(void);
int test_mm256_castsi256_si128(void);
int test_mm_cvtsi32_si128(void);
int test_mm_cvtsi128_si32(void);
int test_mm256_cvtepi32_pd(void);
int test_mm256_cvtepi32_ps(void);
int test_mm256_alignr_epi8();
int test_mm_cmpestri();
int test_mm_cmpestrm();
int test_mm512_div_epi8();
int test_mm512_div_epi16();
int test_mm512_div_epi32();
int test_mm512_div_epi64();
int test_mm512_div_epu8();
int test_mm512_div_epu16();
int test_mm512_div_epu32();
int test_mm512_div_epu64();
int test_mm512_div_round_ps();
int test_mm512_div_round_pd();
int test_mm_insert_epi32();
int test_mm256_insert_epi32();
int test_mm256_insert_epi64();
int test_mm512_castpd128_pd512();
int test_mm512_castpd512_pd128();
int test_mm512_castps128_ps512();
int test_mm512_castps512_ps128();
int test_mm512_cvtepi32_pd();
int test_mm512_cvtepi32_ps();
int test_mm512_insertf32x8();
int test_mm512_insertf64x4();
int test_mm512_inserti32x8();
int test_mm512_inserti64x4();
int test_mm512_permutexvar_epi32();
int test_mm512_permutex2var_epi32();
int test_mm256_cmp_pd();
int test_mm256_cmp_ps();
int test_mm512_cmp_pd_mask();
int test_mm512_cmp_ps_mask();
int test_mm_load_epi32();
int test_mm_load_epi64();
int test_mm_load_si128();
int test_mm_load_pd();
int test_mm_load_ps();
int test_mm256_load_epi32();
int test_mm256_load_epi64();
int test_mm256_load_pd();
int test_mm256_load_ps();
int test_mm512_load_epi32();
int test_mm512_load_epi64();
int test_mm512_load_pd();
int test_mm512_load_ps();
int test_mm_store_epi32();
int test_mm_store_epi64();
int test_mm_store_si128();
int test_mm_store_pd();
int test_mm_store_ps();
int test_mm256_store_epi32();
int test_mm256_store_epi64();
int test_mm256_store_pd();
int test_mm256_store_ps();
int test_mm512_store_epi32();
int test_mm512_store_epi64();
int test_mm512_store_pd();
int test_mm512_store_ps();
int test_mm_max_epu8();
int test_mm_min_epu8();
int test_mm256_max_epi32();
int test_mm512_max_epi32();
int test_mm_packs_epi16();
int test_mm_packs_epi32();
int test_mm256_packs_epi32();
int test_mm512_packs_epi32();
int test_mm_malloc();

#endif
