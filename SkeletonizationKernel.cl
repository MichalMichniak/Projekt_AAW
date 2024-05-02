// __constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 

__kernel void skeletonization1(__read_write image2d_t inputImage, __read_write image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 Gx = (float4)(0);
	float4 Gy = Gx;
	
	if( coord.x >= 1 && coord.x < (get_global_size(0)-1) && coord.y >= 1 && coord.y < get_global_size(1) - 1)
	{

		bool i00 = read_imageui(inputImage, (int2)(coord.x - 1, coord.y + 1)).x > 128;
		bool i10 = read_imageui(inputImage, (int2)(coord.x - 0, coord.y + 1)).x > 128;
		bool i20 = read_imageui(inputImage, (int2)(coord.x + 1, coord.y + 1)).x > 128;
		bool i01 = read_imageui(inputImage, (int2)(coord.x - 1, coord.y + 0)).x > 128;
		bool i11 = read_imageui(inputImage, (int2)(coord.x - 0, coord.y + 0)).x > 128;
		bool i21 = read_imageui(inputImage, (int2)(coord.x + 1, coord.y + 0)).x > 128;
		bool i02 = read_imageui(inputImage, (int2)(coord.x - 1, coord.y - 1)).x > 128;
		bool i12 = read_imageui(inputImage, (int2)(coord.x - 0, coord.y - 1)).x > 128;
		bool i22 = read_imageui(inputImage, (int2)(coord.x + 1, coord.y - 1)).x > 128;

		 int BP = 8 - ((int)i00 + (int)i01 + (int)i02 + (int)i10 + (int)i12 + (int)i20 + (int)i21 + (int)i22); 
		 int AP = (int)(!i01 && i00) + (int)(!i00 && i10) + (int)(!i10 && i20) + (int)(!i20 && i21) + (int)(!i21 && i22) + (int)(!i22 && i12) + (int)(!i12 && i02) + (int)(!i02 && i01);
		// (BP>=2) && (BP<=6) && (AP == 1) && ((i01 && i10 && i21) || (i12 && i10 && i21))
		// 
		
		//int Xh = (int)(!i10 && (i00 || i01)) + (int)(!i01 && (i02 || i12)) + (int)(!i12 && (i22 || i21)) + (int)(!i21 && (i20 || i10));
		//int n1 = (int)(i10 || i00) + (int)(i01 || i02) + (int)(i12 || i22) + (int)(i21 || i20);
		//int n2 = (int)(i01 || i00) + (int)(i12 || i02) + (int)(i21 || i22) + (int)(i10 || i20);
		//int min_n = min(n1,n2);
		//(Xh == 1) && (min_n <= 3) && (2 <= min_n) && ((i00 || i01 || !i20) && i10)
		//if(coord.x == 4322 && coord.y == 5557) printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d \n", BP, AP, i01, i00, i10, i20, i21, i22, i12, i02, i11);
		if((BP>=2) && (BP<=6) && (AP == 1) && (!i01 || !i10 || !i21) && (!i12 || !i10 || !i21) && i11){
			Gx = (float4)(0);//,0,255,255
		}else{
			Gx = convert_float4(read_imageui(inputImage, (int2)(coord.x - 0, coord.y + 0)));
		};
		write_imageui(outputImage, coord, convert_uint4(Gx));
	}

			
}


__kernel void skeletonization2(__read_write image2d_t inputImage, __read_write image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 Gx = (float4)(0);
	float4 Gy = Gx;
	
	if( coord.x >= 1 && coord.x < (get_global_size(0)-1) && coord.y >= 1 && coord.y < get_global_size(1) - 1)
	{
		bool i00 = read_imageui(outputImage, (int2)(coord.x - 1, coord.y + 1)).x > 128;
		bool i10 = read_imageui(outputImage, (int2)(coord.x - 0, coord.y + 1)).x > 128;
		bool i20 = read_imageui(outputImage, (int2)(coord.x + 1, coord.y + 1)).x > 128;
		bool i01 = read_imageui(outputImage, (int2)(coord.x - 1, coord.y + 0)).x > 128;
		bool i11 = read_imageui(outputImage, (int2)(coord.x - 0, coord.y + 0)).x > 128;
		bool i21 = read_imageui(outputImage, (int2)(coord.x + 1, coord.y + 0)).x > 128;
		bool i02 = read_imageui(outputImage, (int2)(coord.x - 1, coord.y - 1)).x > 128;
		bool i12 = read_imageui(outputImage, (int2)(coord.x - 0, coord.y - 1)).x > 128;
		bool i22 = read_imageui(outputImage, (int2)(coord.x + 1, coord.y - 1)).x > 128;
		
		int BP = 8 - ((int)i00 + (int)i01 + (int)i02 + (int)i10 + (int)i12 + (int)i20 + (int)i21 + (int)i22); 
		int AP = (int)(!i01 && i00) + (int)(!i00 && i10) + (int)(!i10 && i20) + (int)(!i20 && i21) + (int)(!i21 && i22) + (int)(!i22 && i12) + (int)(!i12 && i02) + (int)(!i02 && i01);

		int Xh = (int)(!i10 && (i00 || i01)) + (int)(!i01 && (i02 || i12)) + (int)(!i12 && (i22 || i21)) + (int)(!i21 && (i20 || i10));
		int n1 = (int)(i10 || i00) + (int)(i01 || i02) + (int)(i12 || i22) + (int)(i21 || i20);
		int n2 = (int)(i01 || i00) + (int)(i12 || i02) + (int)(i21 || i22) + (int)(i10 || i20);
		int min_n = min(n1,n2);
		//(Xh == 1) && (min_n <= 3) && (2 <= min_n) && ((i22 || i21 || !i02) && i12)
		//if(coord.x == 4322 && coord.y == 5557) printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d \n", BP, AP, i01, i00, i10, i20, i21, i22, i12, i02, i11);
		if((BP>=2) && (BP<=6) && (AP == 1) && (!i12 || !i01 || !i21) && (!i01 || !i10 || !i12) && i11){
			Gx = (float4)(0);//,255,0,255
		}else{
			Gx = convert_float4(read_imageui(outputImage, (int2)(coord.x - 0, coord.y + 0)));
		};
		write_imageui(inputImage, coord, convert_uint4(Gx));
	
	}

			
}

__kernel void skeletonization3(__read_write image2d_t inputImage, __read_write image2d_t refImage, __global int *found)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 Gx = (float4)(0);
	float4 Gy = Gx;
	
	if( coord.x >= 1 && coord.x < (get_global_size(0)-1) && coord.y >= 1 && coord.y < get_global_size(1) - 1)
	{
		Gx = convert_float4(read_imageui(inputImage, (int2)(coord.x - 0, coord.y + 0)));
		if(read_imageui(inputImage, (int2)(coord.x - 0, coord.y + 0)).x != read_imageui(refImage, (int2)(coord.x - 0, coord.y + 0)).x){
			*found = 1;
		}
		
		write_imageui(refImage, coord, read_imageui(inputImage, (int2)(coord.x, coord.y)));
		
	}

			
}

__kernel void skeletonization4(__read_write image2d_t inputImage, __read_write image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 Gx = (float4)(0);
	float4 Gy = Gx;
	
	if( coord.x >= 1 && coord.x < (get_global_size(0)-1) && coord.y >= 1 && coord.y < get_global_size(1) - 1)
	{
		bool i00 = read_imageui(outputImage, (int2)(coord.x - 1, coord.y + 1)).x > 128;
		bool i10 = read_imageui(outputImage, (int2)(coord.x - 0, coord.y + 1)).x > 128;
		bool i20 = read_imageui(outputImage, (int2)(coord.x + 1, coord.y + 1)).x > 128;
		bool i01 = read_imageui(outputImage, (int2)(coord.x - 1, coord.y + 0)).x > 128;
		bool i11 = read_imageui(outputImage, (int2)(coord.x - 0, coord.y + 0)).x > 128;
		bool i21 = read_imageui(outputImage, (int2)(coord.x + 1, coord.y + 0)).x > 128;
		bool i02 = read_imageui(outputImage, (int2)(coord.x - 1, coord.y - 1)).x > 128;
		bool i12 = read_imageui(outputImage, (int2)(coord.x - 0, coord.y - 1)).x > 128;
		bool i22 = read_imageui(outputImage, (int2)(coord.x + 1, coord.y - 1)).x > 128;

		//int BP = 8 - ((int)i00 + (int)i01 + (int)i02 + (int)i10 + (int)i12 + (int)i20 + (int)i21 + (int)i22); 
		//int AP = (int)(!i01 && i00) + (int)(!i00 && i10) + (int)(!i10 && i20) + (int)(!i20 && i21) + (int)(!i21 && i22) + (int)(!i22 && i12) + (int)(!i12 && i02) + (int)(!i02 && i01);

		//int Xh = (int)(!i10 && (i00 || i01)) + (int)(!i01 && (i02 || i12)) + (int)(!i12 && (i22 || i21)) + (int)(!i21 && (i20 || i10));
		//int n1 = (int)(i10 || i00) + (int)(i01 || i02) + (int)(i12 || i22) + (int)(i21 || i20);
		//int n2 = (int)(i01 || i00) + (int)(i12 || i02) + (int)(i21 || i22) + (int)(i10 || i20);
		//int min_n = min(n1,n2);
		//(Xh == 1) && (min_n <= 3) && (2 <= min_n) && ((i22 || i21 || !i02) && i12)
		//if(coord.x == 4322 && coord.y == 5557) printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d \n", BP, AP, i01, i00, i10, i20, i21, i22, i12, i02, i11);
		//if((BP>=2) && (BP<=6) && (AP == 1) && (!i12 || !i01 || !i21) && (!i01 || !i10 || !i12) && i11){
		//	Gx = (float4)(0);//,255,0,255
		//}else{
			Gx = convert_float4(read_imageui(outputImage, (int2)(coord.x - 0, coord.y + 0)));
		//};
		write_imageui(inputImage, coord, convert_uint4(Gx));
		
	}
}

__kernel void skeletonization_check(__read_write image2d_t inputImage, __read_write image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 Gx = (float4)(0);
	float4 Gy = Gx;
	write_imageui(outputImage, coord, read_imageui(inputImage,coord));
}