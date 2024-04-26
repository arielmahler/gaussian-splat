#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_utils.h>
#include <hls_math.h>
#include <ap_fixed.h>

typedef ap_axis<64,1,1,1> AXI_VAL;
typedef ap_axis<32,1,1,1> RET_VAL;
typedef ap_fixed<9, 1, AP_TRN, AP_SAT> frac_t;
typedef ap_fixed<16, 8, AP_TRN, AP_SAT> data_t;
typedef ap_fixed<16, 8, AP_TRN, AP_SAT> tensor_t;
// Placeholder type
typedef long long payload_t;
typedef ap_fixed<16, 8, AP_TRN, AP_SAT> coef_t;

typedef union
{
	float f;
	uint32_t ui;
} input_conversion;

#define WIN_WIDTH 4
#define N_BINS 8

void histogram_tensor (hls::stream<AXI_VAL>& x, hls::stream<RET_VAL>& y) {
#pragma HLS INTERFACE axis register both port=x
#pragma HLS INTERFACE axis register both port=y
#pragma HLS INTERFACE ap_ctrl_none port=return

  // Create histogram_tensor, setting all values to zero
  tensor_t histogram_tensor[WIN_WIDTH + 2][WIN_WIDTH + 2][N_BINS];
#pragma HLS BIND_STORAGE variable=histogram_tensor type=ram_t2p
  for (int i = 0; i < WIN_WIDTH + 2; i++){
	  for(int j = 0; j < WIN_WIDTH + 2; j++){
		  for(int k = 0; k < N_BINS; k++){
			  histogram_tensor[i][j][k] = 0;
		  }
	  }
  }

  // Additional flags
  ap_uint<4> TKEEP_on = 0xf;
  ap_uint<4> TSTRB_on = 0xf;
  ap_uint<4> TKEEP_off = 0x0;
  ap_uint<4> TSTRB_off = 0x0;

  RET_VAL output;

  // All current data should not count as data
  output.keep = TKEEP_off;
  output.strb = TSTRB_off;

  while(1) {
#pragma HLS PIPELINE II=8
	AXI_VAL tmp1;

	// In theory x now contains the 4 values
	// row_bin
	// col_bin
	// magnitude
	// orientation_bin
	x.read(tmp1);

	// Let each of the values exist along 16 bits
	int16_t bitmask = 0xFFFF;
	int16_t sign_bitmask = 0x8000;
	int16_t exp_bitmask = 0x7A00;
	int16_t mantissa_bitmask = 0x03FF;
	input_conversion ic;
	payload_t data = tmp1.data;

	int16_t in_f = (data >> 0) & bitmask;
	ic.ui = ((in_f&sign_bitmask)<<16) | (((in_f&exp_bitmask)+0x1C000)<<13) | ((in_f&mantissa_bitmask)<<13);
	half orientation_bin_h = ic.f;

	in_f = (data >> 16) & bitmask;
	ic.ui = ((in_f&sign_bitmask)<<16) | (((in_f&exp_bitmask)+0x1C000)<<13) | ((in_f&mantissa_bitmask)<<13);
	half magnitude_h = ic.f;

	in_f = (data >> 32) & bitmask;
	ic.ui = ((in_f&sign_bitmask)<<16) | (((in_f&exp_bitmask)+0x1C000)<<13) | ((in_f&mantissa_bitmask)<<13);
	half col_bin_h = ic.f;

	in_f = (data >> 48) & bitmask;
	ic.ui = ((in_f&sign_bitmask)<<16) | (((in_f&exp_bitmask)+0x1C000)<<13) | ((in_f&mantissa_bitmask)<<13);
	half row_bin_h = ic.f;

	data_t row_bin = (data_t) row_bin_h;
	data_t col_bin = (data_t) col_bin_h;
	data_t magnitude = (data_t) magnitude_h;
	data_t orientation_bin = (data_t) orientation_bin_h;

	// split the row_bin and col_bin into their integer and fractional parts
	int8_t row_int = row_bin.to_int();
	frac_t row_fraction = row_bin.to_float() - (float)row_int;
	int8_t col_int = col_bin.to_int();
	frac_t col_fraction = col_bin.to_float() - (float)col_int;
	int8_t orientation_bin_int = orientation_bin.to_int();
	frac_t orientation_fraction = orientation_bin.to_float() - (float)orientation_bin_int;

	if (orientation_bin_int < 0) {
		orientation_bin_int += N_BINS;
	}
	if (orientation_bin_int >= N_BINS) {
		orientation_bin_int -= N_BINS;
	}

	// calculate the coefficients
	coef_t c1 = magnitude * row_fraction;
	coef_t c0 = magnitude * (1 - row_fraction);

	coef_t c11 = c1 * col_fraction;
	coef_t c10 = c1 * (1 - col_fraction);
	coef_t c01 = c0 * col_fraction;
	coef_t c00 = c0 * (1 - col_fraction);

	coef_t c111 = c11 * orientation_fraction;
	coef_t c110 = c11 * (1 - orientation_fraction);
	coef_t c101 = c10 * orientation_fraction;
	coef_t c100 = c10 * (1 - orientation_fraction);
	coef_t c011 = c01 * orientation_fraction;
	coef_t c010 = c01 * (1 - orientation_fraction);
	coef_t c001 = c00 * orientation_fraction;
	coef_t c000 = c00 * (1 - orientation_fraction);
	// Add tensors
	tensor_t read1 = histogram_tensor[row_int + 1][col_int + 1][orientation_bin_int];
	tensor_t read2 = histogram_tensor[row_int + 1][col_int + 1][(orientation_bin_int + 1) % N_BINS];
	tensor_t read3 = histogram_tensor[row_int + 1][col_int + 2][orientation_bin_int];
	tensor_t read4 = histogram_tensor[row_int + 1][col_int + 2][(orientation_bin_int + 1) % N_BINS];
	tensor_t read5 = histogram_tensor[row_int + 2][col_int + 1][orientation_bin_int];
	tensor_t read6 = histogram_tensor[row_int + 2][col_int + 1][(orientation_bin_int + 1) % N_BINS];
	tensor_t read7 = histogram_tensor[row_int + 2][col_int + 2][orientation_bin_int];
	tensor_t read8 = histogram_tensor[row_int + 2][col_int + 2][(orientation_bin_int + 1) % N_BINS];

	histogram_tensor[row_int + 1][col_int + 1][orientation_bin_int] = c000 + read1;
	histogram_tensor[row_int + 1][col_int + 1][(orientation_bin_int + 1) % N_BINS] = c001 + read2;
	histogram_tensor[row_int + 1][col_int + 2][orientation_bin_int] = c010 + read3;
	histogram_tensor[row_int + 1][col_int + 2][(orientation_bin_int + 1) % N_BINS] = c011 + read4;
	histogram_tensor[row_int + 2][col_int + 1][orientation_bin_int] = c100 + read5;
	histogram_tensor[row_int + 2][col_int + 1][(orientation_bin_int + 1) % N_BINS] = c101 + read6;
	histogram_tensor[row_int + 2][col_int + 2][orientation_bin_int] = c110 + read7;
	histogram_tensor[row_int + 2][col_int + 2][(orientation_bin_int + 1) % N_BINS] = c111 + read8;

	output.user = tmp1.user;

	if (tmp1.last) {
		break;
	}
  }

  input_conversion out;
  for (int i = (WIN_WIDTH + 1); i >= 0; i--){
	  for(int j = (WIN_WIDTH + 1); j >= 0; j--){
		  for(int k = (N_BINS - 1); k >= 0; k--){
			  out.f = histogram_tensor[i][j][k].to_float();
			  output.data = out.ui;
			  output.last = (i == 0) && (j == 0) && (k == 0);
			  output.keep = TKEEP_on;
			  output.strb = TSTRB_on;

			  y.write(output);

		  }
	  }
  }
}
