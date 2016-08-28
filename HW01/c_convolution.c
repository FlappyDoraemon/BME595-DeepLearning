void c_convo(double *mylena , double * kernel , double * output , int height , int width , int k_height , int k_width)
{
/*9*9 is the standard parameter of kernel so do not input here*/
    int width_out = width - k_width + 1;
    for(int j = 0 ; j < height - k_height + 1 ; j ++)
    {
        for(int k = 0 ; k < width - k_width + 1 ; k++)
        {
            double sum_temp = 0;
            for(int innerj = 0 ; innerj < k_height ; innerj++)
            {
                for(int innerk = 0 ; innerk < k_width ; innerk++)
                {
                    /*lena_pix(j,j+innerj,k+innerk) .* kernel_pix(innerj , innerk)*/
                    sum_temp += mylena[(j+innerj)*width+(k+innerk)] * kernel[innerj*k_width + innerk];
                }
            }
            output[j * width_out + k] = sum_temp;
        }
    }
}
