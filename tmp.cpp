#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_utils.h>

typedef ap_axis<64,1,1,1> AXI_VAL;
typedef ap_axis<32,1,1,1> RET_VAL;
typedef ap_int<32> uint32_t;
typedef ap_int<16> uint16_t;
typedef ap_fixed<16, 16, AP_TRN, AP_SAT> frac_t;
// TODO: revisit these types
typedef ap_fixed<32, 16, AP_TRN, AP_SAT> data_t;
typedef ap_fixed<32, 16, AP_TRN, AP_SAT> tensor_t;

// Placeholder type
typedef ap_int<64> payload_t;
typedef ap_fixed<32, 16, AP_TRN, AP_SAT> coef_t;

#define WIN_WIDTH 4
#define N_BINS 8

void histogram_tensor (hls::stream<AXI_VAL>& x, hls::stream<RET_VAL>& y) {
#pragma HLS INTERFACE axis register both port=x
#pragma HLS INTERFACE axis register both port=y
#pragma HLS INTERFACE ap_ctrl_none port=return

  // Create histogram_tensor, setting all values to zero
  tensor_t histogram_tensor[WIN_WIDTH + 2][WIN_WIDTH + 2][N_BINS] = {};
#pragma HLS BIND_STORAGE variable=histogram_tensor type=ram_t2p
//#pragma HLS ARRAY_PARTITION dim=1 factor=1 type=block variable=histogram_tensor
//#pragma HLS ARRAY_PARTITION dim=2 factor=1 type=block variable=histogram_tensor
//#pragma HLS ARRAY_PARTITION dim=3 factor=1 type=block  variable=histogram_tensor



  RET_VAL output;

  while(1) {
#pragma HLS PIPELINE II=8
	AXI_VAL tmp1;

	// In theory x now contains the 4 values
	// row_bin
	// col_bin
	// magnitude
	// orientation_bin
	x.read(tmp1);

	// Let each of the values exist along 32 bits
	uint32_t bitmask = 0xFFFFFFFF;
	payload_t data = tmp1.data;
	half row_bin_f = (data >> 0) & bitmask;
	half col_bin_f = (data >> 16) & bitmask;
	half magnitude_f = (data >> 32) & bitmask;
	half orientation_bin_f = (data >> 48) & bitmask;
	data_t row_bin = (data_t) row_bin_f;
	data_t col_bin = (data_t) col_bin_f;
	data_t magnitude = (data_t) magnitude_f;
	data_t orientation_bin = (data_t) orientation_bin_f;

	// split the row_bin and col_bin into their integer and fractional parts
	uint16_t row_int = row_bin >> 16;
	frac_t row_fraction = row_bin & 0xFFFF;
	uint16_t col_int = col_bin >> 16;
	frac_t col_fraction = col_bin & 0xFFFF;
	uint16_t orientation_bin_int = orientation_bin >> 16;
	frac_t orientation_fraction = orientation_bin & 0xFFFF;

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
	data_t read1 = histogram_tensor[row_int + 1][col_int + 1][orientation_bin_int];
	data_t read2 = histogram_tensor[row_int + 1][col_int + 1][(orientation_bin_int + 1) % N_BINS];
	data_t read3 = histogram_tensor[row_int + 1][col_int + 2][orientation_bin_int];
	data_t read4 = histogram_tensor[row_int + 1][col_int + 2][(orientation_bin_int + 1) % N_BINS];
	data_t read5 = histogram_tensor[row_int + 2][col_int + 1][orientation_bin_int];
	data_t read6 = histogram_tensor[row_int + 2][col_int + 1][(orientation_bin_int + 1) % N_BINS];
	data_t read7 = histogram_tensor[row_int + 2][col_int + 2][orientation_bin_int];
	data_t read8 = histogram_tensor[row_int + 2][col_int + 2][(orientation_bin_int + 1) % N_BINS];

//#pragma HLS PROTOCOL mode=fixed

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

  // total area of the array
  int dim1 = (WIN_WIDTH + 2);
  int dim2 = (WIN_WIDTH + 2);
  int dim3 = N_BINS;

  // Additional flags
  ap_uint<4> TKEEP = 0xf;
  ap_uint<4> TSTRB = 0xf;

  while (dim1 >= 0) {
	  output.data = (float)histogram_tensor[dim1][dim2][dim3];
	  output.last = 0;
	  output.keep = TKEEP;
	  output.strb = TSTRB;

	  if (dim3 > 0) {
		  dim3 -= 1;
	  } else if (dim2 > 0) {
		  dim2 -= 1;
		  dim3 = N_BINS;
	  } else if (dim1 > 0) {
		  dim1 -= 1;
		  dim2 = WIN_WIDTH + 2;
		  dim3 = N_BINS;
	  } else { // Exit while loop
		  break;
	  }
	  y.write(output);
  }
  output.data = (float)histogram_tensor[0][0][0];
  output.last = 1;
  y.write(output);
}
