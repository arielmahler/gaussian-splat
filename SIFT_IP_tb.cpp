#include <iostream>
#include <hls_stream.h>
#include <ap_utils.h>
#include <ap_axi_sdata.h>
#include "hls_math.h"

using namespace std;

void histogram_tensor(hls::stream<ap_axis<64,1,1,1> > &A, hls::stream<ap_axis<32,1,1,1> > &B);

int main()
{
  half row=-0.9453;
  half col=3.936;
  half mag=0.2256;
  half bin=0.00256;
  hls::stream<ap_axis<64,1,1,1>> A;
  hls::stream<ap_axis<32,1,1,1>> B;
  ap_axis<64,1,1,1> tmp1;
  ap_axis<32,1,1,1> tmp2;

  ap_uint<64> in = 0xbb9043df3338193e;

  tmp1.data = in;
  cout << tmp1.data << endl;
  tmp1.keep = 1;
  tmp1.strb = 1;
  tmp1.user = 1;
  tmp1.last = 1;
  tmp1.id = 0;
  tmp1.dest = 1;

  A.write(tmp1);
  histogram_tensor(A,B);
  B.read(tmp2);

  float check_val = tmp2.data.to_float();

  if (tmp2.data.to_float() != 105)
  {
    cout << "ERROR: results mismatch" << endl;
    return 1;
  }
    cout << "Success: results match" << endl;
    return 0;
}
