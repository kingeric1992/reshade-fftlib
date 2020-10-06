//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  fft_c2c.fxh header demo usage for ReShade.
//                                              Oct.3.2020 by kingeric1992
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "macro_common.fxh"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Setup
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// setup the getter & input dimension for input matrix
#define FFT_SRC_SIZEX       512
#define FFT_SRC_SIZEY       512
#define FFT_SRC_GET(_x, _y) (tex2Dfetch(sampSrc,int4(_x,_y,0,0)).x)
#define FFT_TRANSFORM_DIR   FFT_INVERSED // FFT_FORWARD

texture2D texSrc    < source  ="fft.jpg"; >
                    { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = R8; }; // src  tex
texture2D texDest   { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = R8; }; // dest tex
sampler2D sampSrc   { Texture = texSrc; FILTER(POINT); };

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Include FFT lib
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "fft_c2c.fxh"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Setup
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

float4 vs_post( uint vid : SV_VERTEXID ) : SV_POSITION {
    return float4((vid.xx == uint2(2,1)? float2(-3,3):float2(1,-1)), 0,1);
}
float ps_post( float4 vpos : SV_POSITION ) : SV_TARGET {
    return fft_c2c::get(vpos.xy).r;
}

// view real component of c2c result.
technique fft_inversed_view {
    pass p0 {
        VertexShader    = vs_post;
        PixelShader     = ps_post;
        RenderTarget    = texDest;
    }
}