/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * test_bl_sgemm.c
 *
 *
 * Purpose:
 * test driver for BLISLAB sgemm routine and reference sgemm routine.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */

//swsok, for mmap
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>

#define PAGE_SIZE (1<<12)
#define PAGE_MASK (~(PAGE_SIZE-1))


#include "bl_sgemm.h"

#define ERROR_TEST

#define TOLERANCE 1E-2
void computeError(
        int    ldc,
        int    ldc_ref,
        int    m,
        int    n,
        float *C,
        float *C_ref
        )
{
    int    i, j;
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
            if ( fabs( C( i, j ) - C_ref( i, j ) ) > TOLERANCE ) {
                printf( "C[ %d ][ %d ] != C_ref, %E, %E\n", i, j, C( i, j ), C_ref( i, j ) );
                break;
            }
        }
    }

}

//swsok, for mmap on /dev/mem
int fid = -1;
off_t offset = 0x200000000LL;

void test_bl_sgemm(
        int m,
        int n,
        int k
        ) 
{
    int    i, j, p, nx;
    float *A, *B, *C, *C_ref;
    float tmp, error, flops;
    float ref_beg, ref_time, bl_sgemm_beg, bl_sgemm_time;
    int    nrepeats;
    int    lda, ldb, ldc, ldc_ref;
    float ref_rectime, bl_sgemm_rectime;

    ssize_t local_offset = 0;
    size_t size;

    if ( fid > 0 ) {
	size = (sizeof(float) * m * k + (PAGE_SIZE-1)) & PAGE_MASK;
	A = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fid, offset);
	local_offset += size;

	size = (sizeof(float) * k * n + (PAGE_SIZE-1)) & PAGE_MASK;
	B = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fid, offset + local_offset);
    } else {
    A    = (float*)malloc( sizeof(float) * m * k );
    B    = (float*)malloc( sizeof(float) * k * n );
    }

    lda = m;
    ldb = k;
    ldc     = m;
    ldc_ref = m;
    C     = bl_malloc_aligned( ldc, n + 4, sizeof(float) );
    C_ref = (float*)malloc( sizeof(float) * m * n );

    nrepeats = 3;

    srand48 (time(NULL));

    // Randonly generate points in [ 0, 1 ].
    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < m; i ++ ) {
            A( i, p ) = (float)( drand48() );	
        }
    }
    for ( j = 0; j < n; j ++ ) {
        for ( p = 0; p < k; p ++ ) {
            B( p, j ) = (float)( drand48() );
        }
    }

    for ( j = 0; j < n; j ++ ) {
        for ( i = 0; i < m; i ++ ) {
            C_ref( i, j ) = (float)( 0.0 );	
                C( i, j ) = (float)( 0.0 );	
        }
    }

    for ( i = 0; i < nrepeats; i ++ ) {
        bl_sgemm_beg = bl_clock();
        {
            bl_sgemm(
                    m,
                    n,
                    k,
                    A,
                    lda,
                    B,
                    ldb,
                    C,
                    ldc
                    );
        }
        bl_sgemm_time = bl_clock() - bl_sgemm_beg;

        if ( i == 0 ) {
            bl_sgemm_rectime = bl_sgemm_time;
        } else {
            bl_sgemm_rectime = bl_sgemm_time < bl_sgemm_rectime ? bl_sgemm_time : bl_sgemm_rectime;
        }
    }

#ifdef ERROR_TEST
    for ( i = 0; i < nrepeats; i ++ ) {
        ref_beg = bl_clock();
        {
            bl_sgemm_ref(
                    m,
                    n,
                    k,
                    A,
                    lda,
                    B,
                    ldb,
                    C_ref,
                    ldc_ref
                    );
        }
        ref_time = bl_clock() - ref_beg;

        if ( i == 0 ) {
            ref_rectime = ref_time;
        } else {
            ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
        }
    }

    computeError(
            ldc,
            ldc_ref,
            m,
            n,
            C,
            C_ref
            );
#endif

    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    printf( "%5d\t %5d\t %5d\t %5.3lf\t %5.3lf\n", 
            m, n, k, flops / bl_sgemm_rectime, flops / ref_rectime );
    fflush(stdout);

    if ( fid > 0 ) {
	size = (sizeof(float) * m * k + (PAGE_SIZE-1)) & PAGE_MASK;
	munmap(A, size);
	size = (sizeof(float) * k * n + (PAGE_SIZE-1)) & PAGE_MASK;
	munmap(B, size);
    } else { 
	    free( A     );
	    free( B     );
    }

    free( C     );
    free( C_ref );
}

int main( int argc, char *argv[] )
{
    int n = 800;
    char *end;

    if (argc < 2) {
	printf("Usage: %s [size, >= 16] </dev/mem> <offset, default=0x200000000>\n", argv[0]);
	return 0;
    }

    if ( argc >= 2 ) {
    	n = atoi(argv[1]);
        if ( n < 16 ) {
	    printf("size(%d) < 16. not valid.\n", n);
	    return 0;
	}
    }

    if ( argc >=3 ) {
	fid = open(argv[2], O_RDWR);
	if (fid < 0) {
	    printf("%s is not opened\n", argv[2]);
	    return 0;
	}
    }

    if ( argc >=4 ) {
	errno = 0;
	offset = strtoull(argv[3], &end, 16);
	if (errno != 0 ) {
	    printf("%s is not valid for offset\n", argv[3]);
	    goto out;
	}
    }
    
    printf("%%m\t%%n\t%%k\t%%MY_GFLOPS\t%%REF_GFLOPS\n");
    for(int i = 16; i <= n; i += 4) {
        test_bl_sgemm( i, i, i );
    }

out:
    if ( fid > 0 ) close (fid);

    return 0;
}

