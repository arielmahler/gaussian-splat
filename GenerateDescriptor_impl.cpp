#include <hls_stream.h>
#include <ap_axi_sdata.h>

typedef ap_axis<128,1,1,1> AXI_VAL;
typedef ap_int<32> uint32_t;
typedef ap_int<16> uint16_t;
typedef ap_fixed<16, 16, AP_TRN, AP_SAT> frac_t;
// TODO: revisit these types
typedef ap_fixed<32, 16, AP_TRN, AP_SAT> data_t;
typedef ap_int<32> tensor_t;

// Placeholder type
typedef ap_int<128> payload_t;
typedef ap_fixed<32, 16, AP_TRN, AP_SAT> coef_t;

// TODO: placeholders for histogram_tensor
#define WIN_WIDTH 11
#define N_BINS 8

void filt (hls::stream<AXI_VAL>& y, hls::stream<AXI_VAL>& x) {
#pragma HLS INTERFACE m_axi depth=11 port=c
#pragma HLS INTERFACE axis register both port=x
#pragma HLS INTERFACE axis register both port=y
#pragma HLS INTERFACE ap_ctrl_none port=return

  while(1) {
	data_t data;
	tensor_t t[WIN_WIDTH][WIN_WIDTH][N_BINS];
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
	data_t row_bin = (payload_t >> 0) & bitmask;
	data_t col_bin = (payload_t >> 32) & bitmask;
	data_t magnitude = (payload_t >> 64) & bitmask;
	data_t orientation_bin = (payload_t >> 96) & bitmask;

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
	t[row_int + 1][col_int + 1][orientation_bin_int] += c000;
	t[row_int + 1][col_int + 1][(orientation_bin_int + 1) % N_BINS] += c001;
	t[row_int + 1][col_int + 2][orientation_bin_int] += c010;
	t[row_int + 1][col_int + 2][(orientation_bin_int + 1) % N_BINS] += c011;
	t[row_int + 2][col_int + 1][orientation_bin_int] += c100;
	t[row_int + 2][col_int + 1][(orientation_bin_int + 1) % N_BINS] += c101;
	t[row_int + 2][col_int + 2][orientation_bin_int] += c110;
	t[row_int + 2][col_int + 2][(orientation_bin_int + 1) % N_BINS] += c111;

	AXI_VAL output;
	output.data = acc;
	output.keep = tmp1.keep;
	output.strb = tmp1.strb;
	output.last = tmp1.last;
	output.dest = tmp1.dest;
	output.id = tmp1.id;
	output.user = tmp1.user;
	y.write(output);

	if (tmp1.last) {
		break;
	}
  }
}
