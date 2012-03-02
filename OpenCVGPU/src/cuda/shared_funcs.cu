/*!
 * @file shared_funcs.cu
 * @author a_hasimoto
 * @date Date Created: 2012/Jan/25
 * @date Last Change: 2012/Feb/17.
 */
#ifndef __SKL_GPU_KERNEL_SHARED_FUNCS_CU__
#define __SKL_GPU_KERNEL_SHARED_FUNCS_CU__


#define _GIdx(elem) ( ( blockIdx.elem * blockDim.elem ) + threadIdx.elem )
#define _SeqIdx(idxx,idxy,width) ( ( (idxy) * (width) ) + (idxx) )

#define GlobalIdx dim3(\
		_GIdx(x) ,\
		_GIdx(y) )
#define inThreadSeqIdx _SeqIdx(threadIdx.x,threadIdx.y,blockDim.x)
#define MEM_WIDTH ( gridDim.x * blockDim.x )
#define MEM_HEIGHT ( gridDim.y * blockDim.y )

#define ThreadSize ( blockDim.x * blockDim.y )


template<class T> __device__ void sumUp(T* array, int length,const size_t idx){
	for(size_t s=length/2;s>0;s>>=1){
		if(idx<s){
			array[idx] += array[idx+s];
		}
		__syncthreads();
	}
}

template<class T> inline __device__ void swap(T& a, T& b){
	{
		T temp(a);
		a = b;
		b = temp;
	}
}


#endif // __SKL_GPU_KERNEL_SHARED_FUNCS_CU__
