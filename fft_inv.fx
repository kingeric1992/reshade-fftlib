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
#define FFT_SRC_SIZEX       256
#define FFT_SRC_SIZEY       256
#define FFT_SRC_GET(_x, _y) (tex2Dfetch(sampSrc,int4(_x,_y,0,0)).x)
#define FFT_SRC_INV(_x, _y) (get(vpos.xy)) // getter for inversed flow

texture2D texSrc    < source  ="fft.jpg"; >
                    { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = R8; }; // src  tex
texture2D texDest   { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = R8; }; // dest tex
texture2D texMid    { Width = FFT_SRC_SIZEX; Height = FFT_SRC_SIZEY; Format = R8; }; // dest tex
sampler2D sampSrc   { Texture = texSrc; FILTER(POINT); };

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Include FFT lib
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// c2c lib support both forward and inverse flow
#include "fft_c2c.fxh"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Shaders
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

float norm( float u) { return log(u + 1)/log(100); }

float4 vs_post( uint vid : SV_VERTEXID ) : SV_POSITION {
    return float4((vid.xx == uint2(2,1)? float2(-3,3):float2(1,-1)),0,1);
}
float ps_mid( float4 vpos : SV_POSITION ) : SV_TARGET {
    vpos.zw  = float2(FFT_SRC_SIZEX, FFT_SRC_SIZEY)*.5;
    vpos.xy += vpos.xy < vpos.zw ? vpos.zw : -vpos.zw;
    return norm(fft_c2c::amp(vpos.xy)); // forward result
}
float ps_post( float4 vpos : SV_POSITION ) : SV_TARGET {
    return norm(fft_c2c::amp(vpos.xy)); // forward result
    //return fft_c2c::getInv(vpos.xy).r;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  Technique
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// view real component of c2c result.
technique fft_c2c_forward_view {
    pass p0 {
        VertexShader    = vs_post;
        PixelShader     = ps_mid;
        RenderTarget    = texMid;
    }
}
technique fft_c2c_inverse_view {
    pass p1 {
        VertexShader    = vs_post;
        PixelShader     = ps_post;
        RenderTarget    = texDest;
    }
}