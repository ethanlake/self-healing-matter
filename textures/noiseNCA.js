/*
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
Usage:
  const gui = new dat.GUI();
  const ca = new CA(gl, models_json, [W, H], gui); // gui is optional
  ca.step();
  
  ca.paint(x, y, radius, modelIndex);
  ca.clearCircle(x, y, radius;

  const stats = ca.benchmark();
  ca.draw();
  ca.draw(zoom);
*/


// Maps the position which is in [-1.0, 1.0]
// to uv which is in [0, 1]
const vs_code = `
    attribute highp vec4 position;
    varying highp vec2 uv;
    void main() {
        uv = position.xy*0.5 + 0.5;
        gl_Position = position;
    }
`

function defInput(name) {
    return `
        uniform Tensor ${name};
        uniform sampler2D ${name}_tex;

        vec4 ${name}_read(vec2 pos, float ch) {return _read(${name}, ${name}_tex, pos, ch);}
        vec4 ${name}_read01(vec2 pos, float ch) {return _read01(${name}, ${name}_tex, pos, ch);}
        vec4 ${name}_readUV(vec2 uv) {return _readUV(${name}, ${name}_tex, uv);}
    `
}

const PREFIX = `
    // #ifdef GL_FRAGMENT_PRECISION_HIGH
    //     precision highp float;
    //     precision highp sampler2D;
    //     precision highp int;
    // #else
    //     precision mediump float;
    //     precision mediump sampler2D;
    //     precision mediump int;
    // #endif
    
    precision highp float;
    precision highp sampler2D;
    precision highp int;
    


    // "Hash without Sine" by David Hoskins (https://www.shadertoy.com/view/4djSRW)
    float hash13(vec3 p3) {
      p3  = fract(p3 * .1031);
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.x + p3.y) * p3.z);
    }
    vec2 hash23(vec3 p3)
    {
        p3 = fract(p3 * vec3(.1031, .1030, .0973));
        p3 += dot(p3, p3.yzx+33.33);
        return fract((p3.xx+p3.yz)*p3.zy);
    }
    
    vec4 hash43(vec3 p){
        vec4 p4 = fract(vec4(p.xyzx)  * vec4(.1031, .1030, .0973, .1099));
        p4 += dot(p4, p4.wzxy+33.33);
        return fract((p4.xxyz+p4.yzzw)*p4.zywx);
    }
    vec4 hash44(vec4 p4) {
        p4 = fract(p4  * vec4(.1031, .1030, .0973, .1099));
        p4 += dot(p4, p4.wzxy+33.33);
        return fract((p4.xxyz+p4.yzzw)*p4.zywx);
    }
    

    struct Tensor {
        vec2 size;
        vec2 gridSize;
        float depth, depth4;
        vec2 packScaleZero;
        
        // tensor.gridSize is the size of the rectangle 
        // containing the channel information of the tensor
        // tensor.size is the spatial size of the original tensor
        // depth: Number of channels
        // depth4: Number of channels // 4
    };
    uniform Tensor u_output;

    vec4 _readUV(Tensor tensor, sampler2D tex, vec2 uv) {
        highp vec4 v = texture2D(tex, uv);

        return v;
    }
    vec2 _getUV(Tensor tensor, vec2 pos, float ch) {
        // pos is the absolute coordinate
        // uv is the texture coordinate
        
        ch += 0.5;
        // [tx, ty] the offset to move to get the desired channel
        float tx = floor(mod(ch, tensor.gridSize.x));
        float ty = floor(ch / tensor.gridSize.x);
        
        #ifdef CIRCULAR_PADDING
            highp vec2 p = fract(pos/tensor.size); // circular padding
            
        #else
            highp vec2 p = clamp(pos, vec2(0.0, 0.0), tensor.size - 0.5); // replicate padding
            p = p / tensor.size;
        #endif 
        
         
        p += vec2(tx, ty); 
        
        p /= tensor.gridSize;
        
        // the output p is in range [0.0, 1.0] 
        
        return p;
    }
    vec4 _read01(Tensor tensor, sampler2D tex, vec2 pos, float ch) {
        // Returns the scaled value of the tensor (between 0.0 and 1.0)
        return texture2D(tex, _getUV(tensor, pos, ch));
    }
    vec4 _read(Tensor tensor, sampler2D tex, vec2 pos, float ch) {
        // Returns the correct value of the tensor
        highp vec2 p = _getUV(tensor, pos, ch);
        return _readUV(tensor, tex, p);
    }
    vec2 getOutputXY() {
        // gl_FragCoord is the coordinate in the texture
        // which contains the tensor information
        // If the original tensor is 3x3 and has 32 channels then
        // The first channel of the texture would look like this
        // 0 0 0 1 1 1 2 2 2 3 3 3
        // 0 0 0 1 1 1 2 2 2 3 3 3
        // 0 0 0 1 1 1 2 2 2 3 3 3
        // 4 4 4 5 5 5 6 6 6 7 7 7 
        // 4 4 4 5 5 5 6 6 6 7 7 7
        // 4 4 4 5 5 5 6 6 6 7 7 7
        
        // Taking the mode with respect to the output size
        // will give us the spatial index in the original tensor

        highp vec2 xy = mod(gl_FragCoord.xy, u_output.size);  
        
        return xy;
        
    }
    float getOutputChannel() {
        highp vec2 xy = floor(gl_FragCoord.xy/u_output.size);
        return xy.y*u_output.gridSize.x+xy.x;
    }

    void setOutput(vec4 v) {
        // highp vec2 p = u_output.packScaleZero;
        // v = v/p.x + p.y;
        // 
        gl_FragColor = v;
    }

    #ifdef SPARSE_UPDATE
        uniform highp sampler2D u_shuffleTex, u_unshuffleTex;
        uniform highp vec2 u_shuffleOfs;
    #endif

    ${defInput('u_input')}

    uniform float u_angle, u_alignment;
    uniform float u_hexGrid;
    uniform highp vec2 HW;
    
    mat2 rotate(float ang) {
        float s = sin(ang), c = cos(ang);
        return mat2(c, s, -s, c);
    }

    vec2 getCellDirection(vec2 xy) {
        vec2 dir = vec2(0.0, 1.0);
        if (u_alignment == 1.0) {
            dir = normalize(xy-0.5*u_input.size);
        } else if (u_alignment == 2.0) {
            vec2 v1 = xy-0.25*u_input.size;
            vec2 v2 = 0.75*u_input.size-xy;
            dir = normalize(v1/pow(length(v1), 3.0) +  v2/pow(length(v2), 3.0));
        }
        dir = rotate(u_angle) * dir;
        return dir;
    }
    // https://www.shadertoy.com/view/Xljczw
    // https://www.shadertoy.com/view/MlXyDl
    // returns xy - in cell pos, zw - skewed cell id
    vec4 getHex(vec2 u) {
        vec2 s = vec2(1., mix(2.0, 1.732, u_hexGrid));
        vec2 p = vec2(0.5*u_hexGrid, 0.5);
        vec2 a = mod(u    ,s)*2.-s;
        vec2 b = mod(u+s*p,s)*2.-s;
        vec2 ai = floor(u/s);
        vec2 bi = floor(u/s+p);
        // skewed coords
        ai = vec2(ai.x-ai.y*u_hexGrid, ai.y*2.0+1.0);
        bi = vec2(bi.x-bi.y*u_hexGrid, bi.y*2.0);
        return dot(a,a)<dot(b,b) ? vec4(a, ai) : vec4(b, bi);    
    }

    vec2 hex2screen(vec2 u) {
        return vec2(u.x + u.y/2.0, u.y*1.732/2.0); 
    }
`;

const PROGRAMS = {
    paint: `
    uniform highp vec2 u_pos;
    uniform float u_r;
    uniform highp vec4 u_brush;
    uniform float u_zoom;

    uniform float u_seed;
    uniform bool u_control;
    uniform float u_noise_level;
    uniform float u_noiseAffectsAlpha;  // 1.0 if noise should affect alpha channel

    void main() {

        vec2 xy = u_pos;
        xy = (xy + u_output.size*(0.5)*(u_zoom-1.0))/u_zoom;
        vec2 xy_out = getOutputXY();
        if (u_hexGrid > 0.0) {
            xy_out = hex2screen(xy_out - u_output.size*0.5);
            xy_out = xy_out + u_output.size*0.5;
        }
        vec2 diff = abs(xy_out-xy);
        if (length(diff)*u_zoom>=u_r)
          discard;

        #ifdef SPARSE_UPDATE
            setOutput(u_brush);
        #else
            if (u_control) {
                setOutput(u_brush);
            } else {
                float chn = getOutputChannel();
                vec4 u = (hash44(vec4(xy_out, chn, u_seed)) - 0.5) * u_noise_level;

                // If noise shouldn't affect alpha, zero out alpha component
                if (u_noiseAffectsAlpha < 0.5) {
                    float baseChIdx = chn * 4.0;
                    // Alpha is channel 3 - check if it's in this texel
                    if (baseChIdx <= 3.0 && baseChIdx + 4.0 > 3.0) {
                        float localIdx = 3.0 - baseChIdx;
                        if (localIdx < 0.5) u.x = 0.0;
                        else if (localIdx < 1.5) u.y = 0.0;
                        else if (localIdx < 2.5) u.z = 0.0;
                        else u.w = 0.0;
                    }
                }

                setOutput(u);
            }
        #endif


    }`,
    perception: `
    const highp mat3 sobelX = mat3(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
    const highp mat3 sobelY = mat3(-1.0,-2.0,-1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);

    const highp mat3 lapX = mat3(0.5, 0.0, 0.5, 2.0, -6.0, 2.0, 0.5, 0.0, 0.5);
    const highp mat3 lapY = mat3(0.5, 2.0, 0.5, 0.0, -6.0, 0.0, 0.5, 2.0, 0.5);

    // Isotropic Laplacian for RD models (normalized by 16, same as Python training)
    const highp mat3 laplacian = mat3(1.0/16.0, 2.0/16.0, 1.0/16.0, 2.0/16.0, -12.0/16.0, 2.0/16.0, 1.0/16.0, 2.0/16.0, 1.0/16.0);

    uniform float u_seed, u_updateProbability;
    uniform float u_dx, u_dy;
    uniform float u_isRD;  // 1.0 for RD models, 0.0 for NCA models
    uniform float u_growingMode;  // 1.0 for Growing NCA models (normalized Sobel filters)
    uniform float u_channelCount;  // Actual number of channels (not padded to multiple of 4)
    uniform float u_isRotInv;  // 1.0 for rotation-invariant NCA models (also set for SQZAST)
    uniform float u_thetaChannel;  // Channel index for theta (default 3)

    vec4 conv3x3(vec2 xy, float inputCh, mat3 filter) {
        highp vec4 a = vec4(0.0);
        for (int y=0; y<3; ++y)
        for (int x=0; x<3; ++x) {
          highp vec2 p = xy+vec2(float(x-1), float(y-1));
          a += filter[y][x] * u_input_read(p, inputCh);
        }
        return a;
    }

    // Read a single channel value from state
    float readChannel(vec2 xy, float rawCh) {
        float ch4 = floor(rawCh / 4.0);
        float component = mod(rawCh, 4.0);
        vec4 v = u_input_read(xy, ch4);
        if (component < 0.5) return v.x;
        else if (component < 1.5) return v.y;
        else if (component < 2.5) return v.z;
        else return v.w;
    }

    // Compute a single perception value for a given flat index
    // For NCA: layout is filter-major [id0,id1,...,idN, sx0,sx1,...,sxN, sy0,...,syN, lap0,...,lapN]
    float computePerceptionValue(vec2 xy, float percIdx, float sobelNorm) {
        float filterBand = floor(percIdx / u_channelCount);
        float rawChannel = mod(percIdx, u_channelCount);
        float ch4 = floor(rawChannel / 4.0);
        float component = mod(rawChannel, 4.0);

        vec4 result;
        if (filterBand < 0.5) {
            // Identity filter
            result = u_input_read(xy, ch4);
            // For rotinv models, zero out the theta channel identity
            if (u_isRotInv > 0.5 && abs(rawChannel - u_thetaChannel) < 0.5) {
                return 0.0;
            }
        } else if (filterBand < 2.5) {
            // Sobel filters
            vec4 dx = conv3x3(xy, ch4, sobelX) / (u_dx * sobelNorm);
            vec4 dy = conv3x3(xy, ch4, sobelY) / (u_dy * sobelNorm);

            if (u_isRotInv > 0.5) {
                // Rotation-invariant: use theta channel for rotation
                // Python: rotated_x = cos_t * sobel_x + sin_t * sobel_y
                //         rotated_y = -sin_t * sobel_x + cos_t * sobel_y
                float theta = readChannel(xy, u_thetaChannel);
                float angle = 2.0 * 3.14159265 * theta;
                float c = cos(angle);
                float s = sin(angle);
                if (filterBand < 1.5) {
                    result = dx * c + dy * s;  // rotated sobel_x
                } else {
                    result = -dx * s + dy * c;  // rotated sobel_y
                }
            } else {
                // Standard NCA: use cell direction
                highp vec2 dir = getCellDirection(xy);
                float s = dir.x, c = dir.y;
                if (filterBand < 1.5) {
                    result = dx * c - dy * s;  // rotated sobel_x
                } else {
                    result = dx * s + dy * c;  // rotated sobel_y
                }
            }
        } else {
            // Laplacian filter
            mat3 lap = lapX / (u_dx * u_dx) + lapY / (u_dy * u_dy);
            result = conv3x3(xy, ch4, lap);
        }

        // Extract the specific component
        if (component < 0.5) return result.x;
        else if (component < 1.5) return result.y;
        else if (component < 2.5) return result.z;
        else return result.w;
    }

    void main() {
        vec2 xy = getOutputXY();


        #ifdef SPARSE_UPDATE
            xy = texture2D(u_shuffleTex, xy/u_output.size).xy+0.5 + u_shuffleOfs;
            xy = mod(xy, u_input.size);

        #endif
        float ch = getOutputChannel();
        if (ch >= u_output.depth4)
            return;

        // RD model: identity and laplacian only (2 filters)
        // Layout is filter-major: [id0,id1,...,idN, lap0,lap1,...,lapN]
        if (u_isRD > 0.5) {
            float filterBand = floor((ch+0.5)/u_input.depth4);
            float inputCh = ch-filterBand*u_input.depth4;
            if (filterBand < 0.5) {
                // Identity filter
                setOutput(u_input_read(xy, inputCh));
            } else {
                // Laplacian filter (unnormalized for rotation invariance)
                highp vec4 res = conv3x3(xy, inputCh, laplacian);
                setOutput(res);
            }
        } else {
            // NCA model: 4 filters per channel
            // Output layout is filter-major: [id0,...,idN, sx0,...,sxN, sy0,...,syN, lap0,...,lapN]
            // Each output texel contains 4 consecutive perception values from this flat layout
            // All NCA models (including Growing NCA) use unnormalized Sobel filters
            float sobelNorm = 1.0;

            // Flat perception index for first component of this texel
            float baseIdx = ch * 4.0;

            // Compute each of the 4 components
            highp vec4 result;
            result.x = computePerceptionValue(xy, baseIdx + 0.0, sobelNorm);
            result.y = computePerceptionValue(xy, baseIdx + 1.0, sobelNorm);
            result.z = computePerceptionValue(xy, baseIdx + 2.0, sobelNorm);
            result.w = computePerceptionValue(xy, baseIdx + 3.0, sobelNorm);

            setOutput(result);
        }
    }`,
    dense: `
    ${defInput('u_control')}
    //u_weightTex contains the layer weights
    uniform sampler2D u_weightTex;
    uniform float u_seed, u_fuzz, u_updateProbability;
    uniform highp vec2 u_weightCoefs; // scale, center
    uniform highp vec2 u_layout;
    uniform highp vec2 grid_size;
    uniform bool bias, pos_emb, relu;
    uniform float u_isRD;  // 1.0 for RD models, 0.0 for NCA models
    
    const float MAX_PACKED_DEPTH = 32.0;
    
    vec4 readWeightUnscaled(vec2 p) {
        highp vec4 w = texture2D(u_weightTex, p);
        return w-u_weightCoefs.y; // centerize
    }
    
    void main() {
      vec2 xy = getOutputXY();
    
    
      // #ifndef SPARSE_UPDATE
      // if (hash13(vec3(xy, u_seed)) > u_updateProbability) {
      //   setOutput(vec4(0.0, 0.0, 0.0, 0.0));
      //   return;
      // }
      // #endif
      
      
      float ch = getOutputChannel();
      if (ch >= u_output.depth4)
          return;


      float d = u_input.depth;
      if (pos_emb) {
        d = d + 2.0;
      }
      if (bias) {
        d = d + 1.0;
      }
      float dy = 1.0 / (d * u_layout.y);
      // float dy = 1.0/(d)/u_layout.y;
      // float dy = 1.0/(u_input.depth+1.0)/u_layout.y;
      
      
      vec2 p = vec2((ch+0.5)/u_output.depth4, dy*0.5);
      vec2 fuzz = (hash23(vec3(xy, u_seed+ch))-0.5)*u_fuzz;
      // vec2 fuzz = vec2(0.0, 0.0);

      vec2 realXY = xy;
      #ifdef SPARSE_UPDATE
        // realXY = texture2D(u_shuffleTex, xy/u_output.size).xy * 255.0 +0.5 + u_shuffleOfs;
        realXY = texture2D(u_shuffleTex, xy/u_output.size).xy + 0.5 + u_shuffleOfs;
        realXY = mod(realXY, HW);
        // realXY = texture2D(u_shuffleTex, xy/u_output.size).xy * (HW.x - 1.0) +0.5 + u_shuffleOfs;
      #endif
      
      
      
      // float modelIdx = u_control_read(realXY+fuzz, 0.0).x+0.5;
      float modelIdx = 0.5;
      
      p.x += floor(mod(modelIdx, u_layout.x));
      p.y += floor(modelIdx/u_layout.x);
      p /= u_layout;
      highp vec4 result = vec4(0.0);
      for (float i=0.0; i < MAX_PACKED_DEPTH; i+=1.0) {
          highp vec4 inVec = u_input_read(xy, i);
          result += inVec.x * readWeightUnscaled(p); p.y += dy;
          result += inVec.y * readWeightUnscaled(p); p.y += dy;
          result += inVec.z * readWeightUnscaled(p); p.y += dy;
          result += inVec.w * readWeightUnscaled(p); p.y += dy;
          if (i+1.5>u_input.depth4) {
              break;
          }
      }
      if (pos_emb) {
        
        highp vec2 pos = floor(realXY);
        highp vec2 delta = vec2(0.5, 0.5) / HW;
        highp vec2 pemb = pos / HW;
        pemb = 2.0 * (pemb - 0.5 + delta);
        pemb = rotate(-u_angle) * pemb;
        result += pemb.y * readWeightUnscaled(p); p.y += dy;
        result += pemb.x * readWeightUnscaled(p); p.y += dy;
      
      };
      if (bias) {
        result += readWeightUnscaled(p);  // bias
        // p.y += dy; 
      };

      result = result*u_weightCoefs.x;
      if (relu) {
        // ReLU activation (same for both NCA and RD models)
        result = max(result, 0.0);
      }
      setOutput(result);
    }`,
    update: `
    ${defInput('u_update')}
    uniform sampler2D u_dropoutMask;
    uniform vec2 u_dropoutMask_size;
    uniform float u_seed, u_updateProbability, u_epsilon;
    uniform float u_dt;
    uniform float u_growingMode;  // 1.0 for Growing NCA (stochastic fire_rate=0.5)
    uniform float u_fireRate;     // Stochastic update rate (default 0.5 for growing mode)
    uniform float u_noiseAffectsAlpha;  // 1.0 if noise should affect alpha channel
    varying vec2 uv;

    #ifdef USE_LIFE_MASK
    // Life mask helper: get alpha from state at any position
    float getAlphaAt(vec2 pos) {
        // Alpha is channel 3 (stored in ch=0, component w for channels 0-3 packed)
        vec4 state0 = u_input_read(pos, 0.0);
        return state0.w;  // Alpha is 4th component of first pack
    }

    // Check if cell is alive: any neighbor in 3x3 has alpha > 0.1
    float getPreLifeMask(vec2 pos) {
        float maxAlpha = 0.0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                maxAlpha = max(maxAlpha, getAlphaAt(pos + vec2(float(dx), float(dy))));
            }
        }
        return step(0.1, maxAlpha);  // 1.0 if any neighbor has alpha > 0.1
    }
    #endif

    void main() {
      vec2 xy = getOutputXY();
      float ch = getOutputChannel();
      highp vec4 state = u_input_read(xy, ch);
      highp vec4 update = vec4(0.0);

      #ifdef USE_LIFE_MASK
      // Compute pre-life mask (before update)
      float preMask = getPreLifeMask(xy);
      #endif

      #ifdef SPARSE_UPDATE
        vec4 shuffleInfo = texture2D(u_unshuffleTex, fract((xy-u_shuffleOfs)/u_output.size));
        if (shuffleInfo.z > 0.5) {
            update = u_update_read(shuffleInfo.xy + 0.5, getOutputChannel());
        }
      #else
        update = u_update_readUV(uv);
      #endif

      // Noise with sqrt(dt) scaling for correct continuum limit (SDE Euler-Maruyama)
      highp vec4 noise = (hash44(vec4(xy, ch, u_seed)) - 0.5) * 2.0 * u_epsilon * sqrt(u_dt);

      // If noise shouldn't affect alpha, zero out alpha component
      if (u_noiseAffectsAlpha < 0.5) {
          float baseChIdx = ch * 4.0;
          // Alpha is channel 3 - check if it's in this texel
          if (baseChIdx <= 3.0 && baseChIdx + 4.0 > 3.0) {
              float localIdx = 3.0 - baseChIdx;
              if (localIdx < 0.5) noise.x = 0.0;
              else if (localIdx < 1.5) noise.y = 0.0;
              else if (localIdx < 2.5) noise.z = 0.0;
              else noise.w = 0.0;
          }
      }

      highp vec4 newState = state + u_dt * update + noise;

      // Growing NCA: stochastic cell updates (fire_rate = 0.5)
      if (u_growingMode > 0.5) {
          float fireRoll = hash13(vec3(xy, u_seed));
          if (fireRoll > u_fireRate) {
              newState = state;  // This cell doesn't fire
          }
      }

      #ifdef USE_LIFE_MASK
      // Apply life mask: cells die if no living neighbors (alpha > 0.1)
      newState = newState * preMask;
      #endif

      // Apply channel dropout mask
      float ch4 = ch;
      vec4 maskVec = texture2D(u_dropoutMask, vec2((ch4 + 0.5) / u_dropoutMask_size.x, 0.5));
      newState = newState * maskVec;

      setOutput(newState);
    }`,
    sqzastThetaUpdate: `
    // SQZAST fixed theta dynamics update
    // Reads from u_input (original state with noise), computes SQZAST theta
    // Reads from u_newState (post-learned-update) for all other channels
    // Combines them so theta uses SQZAST rule, other channels use learned update
    ${defInput('u_newState')}
    uniform float u_seed;
    uniform float u_dt;
    uniform float u_thetaChannel;  // Which channel is theta (default 3)
    varying vec2 uv;

    void main() {
      vec2 xy = getOutputXY();
      float ch = getOutputChannel();

      // Determine which raw channel indices are in this texel
      float baseChIdx = ch * 4.0;

      // Read post-learned-update state for non-theta channels
      highp vec4 learnedState = u_newState_read(xy, ch);
      // Read original state (pre-learned-update) for theta calculations
      highp vec4 originalState = u_input_read(xy, ch);

      highp vec4 newState = learnedState;  // Start with learned update

      // Check if any of the 4 components in this texel is the theta channel
      for (float i = 0.0; i < 4.0; i += 1.0) {
        float rawCh = baseChIdx + i;

        // Only process if this is the theta channel
        if (abs(rawCh - u_thetaChannel) < 0.5) {
          // Get original theta value at this position (before learned update)
          float theta;
          if (i < 0.5) theta = originalState.x;
          else if (i < 1.5) theta = originalState.y;
          else if (i < 2.5) theta = originalState.z;
          else theta = originalState.w;

          // SQZAST update rule:
          // 1. Random binary mask (probability 0.5) - decides if this site updates
          float mask = step(0.5, hash13(vec3(xy, u_seed)));

          // 2. Random direction: 1=N, 2=S, 3=E, 4=W
          float dir = floor(hash13(vec3(xy, u_seed + 100.0)) * 4.0) + 1.0;

          // Get neighbor theta values from ORIGINAL state using periodic boundary
          // North: y-1, South: y+1, East: x+1, West: x-1
          float ch4_theta = floor(u_thetaChannel / 4.0);
          float comp_theta = mod(u_thetaChannel, 4.0);

          vec4 northVal = u_input_read(xy + vec2(0.0, -1.0), ch4_theta);
          vec4 southVal = u_input_read(xy + vec2(0.0, 1.0), ch4_theta);
          vec4 eastVal = u_input_read(xy + vec2(1.0, 0.0), ch4_theta);
          vec4 westVal = u_input_read(xy + vec2(-1.0, 0.0), ch4_theta);

          float theta_north, theta_south, theta_east, theta_west;
          if (comp_theta < 0.5) {
            theta_north = northVal.x; theta_south = southVal.x;
            theta_east = eastVal.x; theta_west = westVal.x;
          } else if (comp_theta < 1.5) {
            theta_north = northVal.y; theta_south = southVal.y;
            theta_east = eastVal.y; theta_west = westVal.y;
          } else if (comp_theta < 2.5) {
            theta_north = northVal.z; theta_south = southVal.z;
            theta_east = eastVal.z; theta_west = westVal.z;
          } else {
            theta_north = northVal.w; theta_south = southVal.w;
            theta_east = eastVal.w; theta_west = westVal.w;
          }

          // Compute dtheta for each direction
          float dtheta_north = theta_north - theta;
          float dtheta_south = theta_south - theta;
          float dtheta_east = theta_east - theta;
          float dtheta_west = theta_west - theta;

          // step function: step(x) = 0 if x < 0, 1 otherwise
          // For N,S: use step(-sin(2*pi*dtheta)) -> sin(2*pi*dtheta) <= 0
          // For E,W: use step(+sin(2*pi*dtheta)) -> sin(2*pi*dtheta) >= 0
          // Note: theta is in [0,1] representing [0, 2*pi] radians
          float two_pi = 6.283185307179586;
          float step_north = step(0.0, -sin(two_pi * dtheta_north));  // step(-sin(2*pi*dtheta))
          float step_south = step(0.0, -sin(two_pi * dtheta_south));  // step(-sin(2*pi*dtheta))
          float step_east = step(0.0, sin(two_pi * dtheta_east));     // step(+sin(2*pi*dtheta))
          float step_west = step(0.0, sin(two_pi * dtheta_west));     // step(+sin(2*pi*dtheta))

          // Compute update for each direction
          float update_north = u_dt * dtheta_north * step_north;
          float update_south = u_dt * dtheta_south * step_south;
          float update_east = u_dt * dtheta_east * step_east;
          float update_west = u_dt * dtheta_west * step_west;

          // Select update based on random direction
          float update_val = 0.0;
          if (dir < 1.5) update_val = update_north;       // N
          else if (dir < 2.5) update_val = update_south;  // S
          else if (dir < 3.5) update_val = update_east;   // E
          else update_val = update_west;                   // W

          // Apply mask and update theta (from original state, not learned state)
          float new_theta = theta + mask * update_val;

          // Write back to appropriate component (overwriting learned theta)
          if (i < 0.5) newState.x = new_theta;
          else if (i < 1.5) newState.y = new_theta;
          else if (i < 2.5) newState.z = new_theta;
          else newState.w = new_theta;
        }
      }

      setOutput(newState);
    }`,
    blendedUpdate: `
    ${defInput('u_update')}
    ${defInput('u_altUpdate')}
    uniform sampler2D u_dropoutMask;
    uniform vec2 u_dropoutMask_size;
    uniform float u_seed, u_updateProbability, u_epsilon;
    uniform float u_dt;
    uniform float u_blendFactor;
    uniform float u_growingMode;  // 1.0 for Growing NCA (stochastic fire_rate=0.5)
    uniform float u_fireRate;     // Stochastic update rate (default 0.5 for growing mode)
    uniform float u_noiseAffectsAlpha;  // 1.0 if noise should affect alpha channel
    varying vec2 uv;

    #ifdef USE_LIFE_MASK
    float getAlphaAt(vec2 pos) {
        vec4 state0 = u_input_read(pos, 0.0);
        return state0.w;
    }
    float getPreLifeMask(vec2 pos) {
        float maxAlpha = 0.0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                maxAlpha = max(maxAlpha, getAlphaAt(pos + vec2(float(dx), float(dy))));
            }
        }
        return step(0.1, maxAlpha);
    }
    #endif

    void main() {
      vec2 xy = getOutputXY();
      float ch = getOutputChannel();
      highp vec4 state = u_input_read(xy, ch);

      #ifdef USE_LIFE_MASK
      float preMask = getPreLifeMask(xy);
      #endif

      highp vec4 update1 = u_update_readUV(uv);
      highp vec4 update2 = u_altUpdate_readUV(uv);
      highp vec4 blendedUpdate = mix(update1, update2, u_blendFactor);

      highp vec4 noise = (hash44(vec4(xy, ch, u_seed)) - 0.5) * 2.0 * u_epsilon * sqrt(u_dt);

      // If noise shouldn't affect alpha, zero out alpha component
      if (u_noiseAffectsAlpha < 0.5) {
          float baseChIdx = ch * 4.0;
          if (baseChIdx <= 3.0 && baseChIdx + 4.0 > 3.0) {
              float localIdx = 3.0 - baseChIdx;
              if (localIdx < 0.5) noise.x = 0.0;
              else if (localIdx < 1.5) noise.y = 0.0;
              else if (localIdx < 2.5) noise.z = 0.0;
              else noise.w = 0.0;
          }
      }

      highp vec4 newState = state + u_dt * blendedUpdate + noise;

      // Growing NCA: stochastic cell updates
      if (u_growingMode > 0.5) {
          float fireRoll = hash13(vec3(xy, u_seed));
          if (fireRoll > u_fireRate) {
              newState = state;
          }
      }

      #ifdef USE_LIFE_MASK
      newState = newState * preMask;
      #endif

      // Apply channel dropout mask
      float ch4 = ch;
      vec4 maskVec = texture2D(u_dropoutMask, vec2((ch4 + 0.5) / u_dropoutMask_size.x, 0.5));
      newState = newState * maskVec;

      setOutput(newState);
    }`,
    rdUpdate: `
    // RD-specific update: combines reaction term (from FC) with diffusion term (laplacian)
    ${defInput('u_update')}
    uniform sampler2D u_dropoutMask;
    uniform vec2 u_dropoutMask_size;
    uniform sampler2D u_diffCoefTex;  // Diffusion coefficients texture
    uniform vec2 u_diffCoefTex_size;  // Size of diff coef texture
    uniform float u_seed, u_updateProbability, u_epsilon;
    uniform float u_dt;
    uniform float u_reactionRate;    // Reaction rate scaling
    uniform float u_diffusionRate;   // Diffusion rate scaling
    varying vec2 uv;

    // Read diffusion coefficient from texture
    float getDiffCoef(float channelIdx) {
        // Diffusion coefficients are stored in a 1D texture, 4 values per RGBA pixel
        float pixelIdx = floor(channelIdx / 4.0);
        float componentIdx = mod(channelIdx, 4.0);
        vec4 pixel = texture2D(u_diffCoefTex, vec2((pixelIdx + 0.5) / u_diffCoefTex_size.x, 0.5));
        if (componentIdx < 0.5) return pixel.r;
        else if (componentIdx < 1.5) return pixel.g;
        else if (componentIdx < 2.5) return pixel.b;
        else return pixel.a;
    }

    // Isotropic Laplacian (normalized by 16, same as Python training)
    vec4 computeLaplacian(vec2 xy, float ch) {
        highp vec4 sum = vec4(0.0);
        // Laplacian kernel weights (normalized)
        // [[1, 2, 1], [2, -12, 2], [1, 2, 1]] / 16
        sum += u_input_read(xy + vec2(-1.0, -1.0), ch) * (1.0/16.0);
        sum += u_input_read(xy + vec2( 0.0, -1.0), ch) * (2.0/16.0);
        sum += u_input_read(xy + vec2( 1.0, -1.0), ch) * (1.0/16.0);
        sum += u_input_read(xy + vec2(-1.0,  0.0), ch) * (2.0/16.0);
        sum += u_input_read(xy + vec2( 0.0,  0.0), ch) * (-12.0/16.0);
        sum += u_input_read(xy + vec2( 1.0,  0.0), ch) * (2.0/16.0);
        sum += u_input_read(xy + vec2(-1.0,  1.0), ch) * (1.0/16.0);
        sum += u_input_read(xy + vec2( 0.0,  1.0), ch) * (2.0/16.0);
        sum += u_input_read(xy + vec2( 1.0,  1.0), ch) * (1.0/16.0);
        return sum;
    }

    void main() {
      vec2 xy = getOutputXY();
      float ch = getOutputChannel();
      highp vec4 state = u_input_read(xy, ch);

      // Reaction term from FC layers
      highp vec4 reaction = u_update_readUV(uv);

      // Diffusion term: laplacian * per-channel diffusion coefficient
      highp vec4 laplacian = computeLaplacian(xy, ch);
      // Apply per-channel diffusion coefficients from texture
      float baseChIdx = ch * 4.0;
      highp vec4 diffCoefs = vec4(
          getDiffCoef(baseChIdx),
          getDiffCoef(baseChIdx + 1.0),
          getDiffCoef(baseChIdx + 2.0),
          getDiffCoef(baseChIdx + 3.0)
      );
      highp vec4 diffusion = laplacian * diffCoefs * u_diffusionRate;

      // RD update: state + dt * (diffusion + reaction * reactionRate)
      // Noise with sqrt(dt) scaling for correct continuum limit (SDE Euler-Maruyama)
      highp vec4 noise = (hash44(vec4(xy, ch, u_seed)) - 0.5) * 2.0 * u_epsilon * sqrt(u_dt);
      highp vec4 newState = state + u_dt * (diffusion + reaction * u_reactionRate) + noise;

      // Apply channel dropout mask
      float ch4 = ch;
      vec4 maskVec = texture2D(u_dropoutMask, vec2((ch4 + 0.5) / u_dropoutMask_size.x, 0.5));
      newState = newState * maskVec;

      setOutput(newState);
    }`,
    blendedRdUpdate: `
    // Blended RD update: blends two reaction terms with shared diffusion
    ${defInput('u_update')}
    ${defInput('u_altUpdate')}
    uniform sampler2D u_dropoutMask;
    uniform vec2 u_dropoutMask_size;
    uniform sampler2D u_diffCoefTex;  // Diffusion coefficients texture
    uniform vec2 u_diffCoefTex_size;  // Size of diff coef texture
    uniform float u_seed, u_updateProbability, u_epsilon;
    uniform float u_dt;
    uniform float u_reactionRate;    // Reaction rate scaling
    uniform float u_diffusionRate;   // Diffusion rate scaling
    uniform float u_blendFactor;     // Blend factor between two models
    varying vec2 uv;

    // Read diffusion coefficient from texture
    float getDiffCoef(float channelIdx) {
        float pixelIdx = floor(channelIdx / 4.0);
        float componentIdx = mod(channelIdx, 4.0);
        vec4 pixel = texture2D(u_diffCoefTex, vec2((pixelIdx + 0.5) / u_diffCoefTex_size.x, 0.5));
        if (componentIdx < 0.5) return pixel.r;
        else if (componentIdx < 1.5) return pixel.g;
        else if (componentIdx < 2.5) return pixel.b;
        else return pixel.a;
    }

    // Isotropic Laplacian (normalized by 16, same as Python training)
    vec4 computeLaplacian(vec2 xy, float ch) {
        highp vec4 sum = vec4(0.0);
        sum += u_input_read(xy + vec2(-1.0, -1.0), ch) * (1.0/16.0);
        sum += u_input_read(xy + vec2( 0.0, -1.0), ch) * (2.0/16.0);
        sum += u_input_read(xy + vec2( 1.0, -1.0), ch) * (1.0/16.0);
        sum += u_input_read(xy + vec2(-1.0,  0.0), ch) * (2.0/16.0);
        sum += u_input_read(xy + vec2( 0.0,  0.0), ch) * (-12.0/16.0);
        sum += u_input_read(xy + vec2( 1.0,  0.0), ch) * (2.0/16.0);
        sum += u_input_read(xy + vec2(-1.0,  1.0), ch) * (1.0/16.0);
        sum += u_input_read(xy + vec2( 0.0,  1.0), ch) * (2.0/16.0);
        sum += u_input_read(xy + vec2( 1.0,  1.0), ch) * (1.0/16.0);
        return sum;
    }

    void main() {
      vec2 xy = getOutputXY();
      float ch = getOutputChannel();
      highp vec4 state = u_input_read(xy, ch);

      // Blend reaction terms from both models
      highp vec4 reaction1 = u_update_readUV(uv);
      highp vec4 reaction2 = u_altUpdate_readUV(uv);
      highp vec4 reaction = mix(reaction1, reaction2, u_blendFactor);

      // Diffusion term: laplacian * per-channel diffusion coefficient
      highp vec4 laplacian = computeLaplacian(xy, ch);
      float baseChIdx = ch * 4.0;
      highp vec4 diffCoefs = vec4(
          getDiffCoef(baseChIdx),
          getDiffCoef(baseChIdx + 1.0),
          getDiffCoef(baseChIdx + 2.0),
          getDiffCoef(baseChIdx + 3.0)
      );
      highp vec4 diffusion = laplacian * diffCoefs * u_diffusionRate;

      // RD update with blended reaction: state + dt * (diffusion + reaction * reactionRate)
      // Noise with sqrt(dt) scaling for correct continuum limit (SDE Euler-Maruyama)
      highp vec4 noise = (hash44(vec4(xy, ch, u_seed)) - 0.5) * 2.0 * u_epsilon * sqrt(u_dt);
      highp vec4 newState = state + u_dt * (diffusion + reaction * u_reactionRate) + noise;

      // Apply channel dropout mask
      float ch4 = ch;
      vec4 maskVec = texture2D(u_dropoutMask, vec2((ch4 + 0.5) / u_dropoutMask_size.x, 0.5));
      newState = newState * maskVec;

      setOutput(newState);
    }`,
    vis: `
    uniform float u_raw;
    uniform float u_zoom;
    uniform float u_perceptionCircle, u_arrows;
    uniform float u_viewChannel;  // -1 = RGB, -2 = grayscale, 0+ = hidden channel index
    uniform float u_invertColors;  // 1.0 = invert, 0.0 = normal
    uniform float u_isRotInv;  // 1.0 for rotation-invariant NCA models
    uniform float u_thetaChannel;  // Channel index for theta (default 3)
    uniform float u_growingMode;  // 1.0 for Growing NCA models (use alpha for blending)
    uniform float u_visScale;     // Scale for auto-scaling visualization
    uniform float u_visOffset;    // Offset for auto-scaling visualization
    varying vec2 uv;

    float clip01(float x) {
        return min(max(x, 0.0), 1.0);
    }

    const float PI = 3.141592653;

    // HSV to RGB conversion
    vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }

    float peak(float x, float r) {
        float y = x/r;
        return exp(-y*y);
    }

    float getElement(vec4 v, float i) {
        if (i<1.0) return v.x;
        if (i<2.0) return v.y;
        if (i<3.0) return v.z;
        return v.w;
    }

    vec3 onehot3(float i) {
        if (i<1.0) return vec3(1.0, 0.0, 0.0);
        if (i<2.0) return vec3(0.0, 1.0, 0.0);
        return vec3(0.0, 0.0, 1.0);
    }

    float sdTriangleIsosceles( in vec2 p, in vec2 q ) {
        p.x = abs(p.x);
        vec2 a = p - q*clamp( dot(p,q)/dot(q,q), 0.0, 1.0 );
        vec2 b = p - q*vec2( clamp( p.x/q.x, 0.0, 1.0 ), 1.0 );
        float s = -sign( q.y );
        vec2 d = min( vec2( dot(a,a), s*(p.x*q.y-p.y*q.x) ),
                      vec2( dot(b,b), s*(p.y-q.y)  ));
        return -sqrt(d.x)*sign(d.y);
    }
    

    void main() {
        vec2 xy = vec2(uv.x, 1.0-uv.y);
        if (u_raw > 0.5) {
            // Show groups of 4 channels
            if (u_input.depth == 12.0) {
                vec3 rgb = vec3(0.0, 0.0, 0.0);
                if (xy.x <= 0.5 || xy.y <= 0.5) {
                    rgb = texture2D(u_input_tex, xy).rgb;
                    

                } else {
                    
                    float r = texture2D(u_input_tex, xy - vec2(0.5, 0.5)).a;
                    float g = texture2D(u_input_tex, xy - vec2(0.5, 0.0)).a;
                    float b = texture2D(u_input_tex, xy - vec2(0.0, 0.5)).a;
                    rgb = vec3(r, g, b);

                }
                
                

                rgb = clamp(rgb + 0.5, 0.0, 1.0);

                
                gl_FragColor = vec4(rgb, 1.0);
 
            } else {
                gl_FragColor = texture2D(u_input_tex, xy);
                gl_FragColor = clamp(gl_FragColor + 0.5, 0.0, 1.0);

                gl_FragColor.a = 1.0;
            } 
        } else {
            xy = (xy + vec2(0.5)*(u_zoom-1.0))/u_zoom;
            xy *= u_input.size;
            vec2 fp = 2.0 * fract(xy)-1.0;

            if (u_hexGrid > 0.0) {
                vec4 r = getHex(xy-u_input.size*0.5);
                xy = r.zw+u_input.size*0.5;
                fp = r.xy;
            }

            vec4 cellRGBA = u_input_read(xy, 0.0);
            vec3 cellRGB;

            // For Growing NCA models:
            // These models store values in [0, 1] range directly (NO +0.5 offset)
            // The notebook display formula is: output = clamp(1.0 - alpha + rgb, 0, 1)
            // where rgb is premultiplied by alpha during training
            if (u_growingMode > 0.5) {
                // Apply auto-scale if enabled (u_visScale != 1.0 or u_visOffset != 0.0)
                vec4 rgba = cellRGBA * u_visScale + u_visOffset;
                rgba = clamp(rgba, 0.0, 1.0);
                // Apply to_rgb formula: output = 1.0 - alpha + rgb
                cellRGB = clamp(vec3(1.0 - rgba.a) + rgba.rgb, 0.0, 1.0);
            } else {
                // Standard texture NCA: apply scale/offset (default: scale=1, offset=0.5)
                cellRGB = clamp(cellRGBA.rgb * u_visScale + u_visOffset, 0.0, 1.0);
            }

            vec3 rgb;
            if (u_viewChannel < -1.5) {
                // Grayscale mode: average of RGB
                float gray = (cellRGB.r + cellRGB.g + cellRGB.b) / 3.0;
                rgb = vec3(gray);
            } else if (u_viewChannel < -0.5) {
                // RGB mode (default)
                rgb = cellRGB;
            } else {
                // Hidden channel mode: show specific channel as grayscale
                float chIdx = floor(u_viewChannel + 0.5);
                float ch4 = floor(chIdx / 4.0);
                float chOffset = mod(chIdx, 4.0);
                vec4 channelData = u_input_read(xy, ch4);
                float val = getElement(channelData, chOffset);

                // For rotinv models viewing theta channel, show as HSV color wheel
                if (u_isRotInv > 0.5 && abs(chIdx - u_thetaChannel) < 0.5) {
                    // theta is in range [-1, 1] (overflow loss boundary), map to hue [0, 1]
                    // val * 0.5 + 0.5 maps [-1,1] -> [0,1]
                    float hue = clamp(val * 0.5 + 0.5, 0.0, 1.0);
                    rgb = hsv2rgb(vec3(hue, 1.0, 1.0));
                } else {
                    // Apply auto-scale: val * scale + offset
                    val = clamp(val * u_visScale + u_visOffset, 0.0, 1.0);
                    rgb = vec3(val);
                }
            }
            
            vec3 cellRGBOriginal = cellRGB;
            cellRGB = rgb;
            if (3.0 < u_zoom) {
                vec2 dir = getCellDirection(floor(xy)+0.5);
                float s = dir.x, c = dir.y;
                fp = mat2(c, s, -s, c) * fp;    
                float r = length(fp);
                float fade = clip01((u_zoom-3.0)/3.0);
                float m = 1.0;//1.0-min(r*r*r, 1.0)*fade;
                rgb *= m;
                if (12.0 < u_zoom) {
                    float ang = atan(-fp.x, fp.y)/(2.0*PI)+0.5;
                    float ch = mod(ang*u_input.depth+1.5, u_input.depth);
                    float barLengh = 0.0;
                    vec3 barColor = vec3(0.5);
                    if (ch < 3.0) {
                        vec3 i3 = onehot3(ch);
                        barColor = i3;
                        barLengh = dot(cellRGBOriginal, i3);
                    } else {
                        vec4 v4 = u_input_read01(xy, floor(ch/4.0));
                        barLengh = getElement(v4, mod(ch, 4.0));
                    }

                    float c = mod(ch, 1.0);
                    c = peak(c-0.5, 0.2);
                    if (r>barLengh)
                      c = 0.0;
                    float fade = clip01((u_zoom-12.0)/8.0);
                    c *= fade;
                    rgb += barColor*c;

                    float arrow = sdTriangleIsosceles((fp+vec2(0.0, 0.95))*vec2(4.0, 4.0), vec2(1.0, 1.0));
                    arrow = clip01(1.0-abs(arrow)*u_zoom/4.0);
                    rgb += arrow*fade*u_arrows;

                    float cr = length(u_input.size/2.0-0.5-xy);
                    rgb += peak(cr-1.5, 0.5/u_zoom)*fade*u_perceptionCircle;
                }
            } 

            rgb = clamp(rgb, 0.0, 1.0);
            if (u_invertColors > 0.5) {
                rgb = vec3(1.0) - rgb;
            }
            gl_FragColor = vec4(rgb, 1.0);
        }
    }`
}

function createPrograms(gl, defines) {
    defines = defines || '';
    const res = {};
    for (const name in PROGRAMS) {
        const fs_code = defines + PREFIX + PROGRAMS[name];
        // vs : vertex shader
        // fs: fragment shader
        const progInfo = twgl.createProgramInfo(gl, [vs_code, fs_code]);
        progInfo.name = name;
        res[name] = progInfo;
    }
    // res is a dictionary of shader programs
    return res;
}

function createTensor(gl, w, h, depth, packScaleZero, is_float = false) {
    // Pack the depth dimension into the spatial dimension
    const depth4 = Math.ceil(depth / 4);
    const gridW = Math.ceil(Math.sqrt(depth4));
    const gridH = Math.floor((depth4 + gridW - 1) / gridW);
    const texW = w * gridW, texH = h * gridH;

    // const ext = gl.getExtension('EXT_color_buffer_float');


    const attachments = [{
        minMag: gl.NEAREST,
        format: gl.RGBA,
        internalFormat: gl.RGBA32F
    }];
    if (is_float) {
        attachments[0].type = gl.FLOAT;
    }

    const fbi = twgl.createFramebufferInfo(gl, attachments, texW, texH);
    const tex = fbi.attachments[0];
    return {
        _type: 'tensor',
        fbi, w, h, depth, gridW, gridH, depth4, tex, packScaleZero,
    };
}

function setTensorUniforms(uniforms, name, tensor) {
    uniforms[name + '.size'] = [tensor.w, tensor.h];
    uniforms[name + '.gridSize'] = [tensor.gridW, tensor.gridH];
    uniforms[name + '.depth'] = tensor.depth;
    uniforms[name + '.depth4'] = tensor.depth4;
    uniforms[name + '.packScaleZero'] = tensor.packScaleZero;
    if (name != 'u_output') {
        uniforms[name + '_tex'] = tensor.tex;
    }
}

function createDenseInfo(gl, params) {
    // params is basically one of layers from the json file

    const center = "center" in params ? params.center : 127.0 / 255.0;

    const [in_n, out_n] = params.shape;
    const info = {
        layout: params.layout, out_n,
        // quantScaleZero: params.quant_scale_zero,
        ready: false
    };

    info.pos_emb = params.pos_emb ? "pos_emb" in params : false;
    info.bias = params.bias;
    var ch_in = in_n;
    ch_in = info.pos_emb ? ch_in - 2 : ch_in;
    ch_in = info.bias ? ch_in - 1 : ch_in;
    info.in_n = ch_in;
    info.coefs = [params.scale, center];

    if ("data_flatten" in params) {
        let width = params.data_shape[1];
        let height = params.data_shape[0];
        info.tex = twgl.createTexture(gl, {
            minMag: gl.NEAREST, src: params.data_flatten, flipY: false, premultiplyAlpha: false,
            width: width, height: height,
            internalFormat: gl.RGBA32F,
            format: gl.RGBA,
            type: gl.FLOAT,
        })
        info.ready = true;

    } else {
        info.tex = twgl.createTexture(gl, {
            minMag: gl.NEAREST, src: params.data, flipY: false, premultiplyAlpha: false,
            format: gl.RGBA,
            internalFormat: gl.RGBA32F,
            type: gl.FLOAT,
        }, () => {
            info.ready = true;
        });
    }
    return info;
}

function createDropoutMaskTexture(gl, channelCount, mask) {
    // Pack mask into RGBA texture (4 channels per pixel)
    const width = Math.ceil(channelCount / 4);
    const data = new Float32Array(width * 4);

    for (let i = 0; i < channelCount; i++) {
        data[i] = mask[i];
    }
    // Pad remaining with 1.0 (active)
    for (let i = channelCount; i < width * 4; i++) {
        data[i] = 1.0;
    }

    return twgl.createTexture(gl, {
        minMag: gl.NEAREST,
        src: data,
        width: width,
        height: 1,
        internalFormat: gl.RGBA32F,
        format: gl.RGBA,
        type: gl.FLOAT,
    });
}

function createDiffCoefTexture(gl, diffCoefArray) {
    // Pack diffusion coefficients into RGBA texture (4 channels per pixel)
    // diffCoefArray is an array of per-channel diffusion coefficients
    const channelCount = diffCoefArray.length;
    const width = Math.ceil(channelCount / 4);
    const data = new Float32Array(width * 4);

    for (let i = 0; i < channelCount; i++) {
        data[i] = diffCoefArray[i];
    }
    // Pad remaining with 0.0
    for (let i = channelCount; i < width * 4; i++) {
        data[i] = 0.0;
    }

    return twgl.createTexture(gl, {
        minMag: gl.NEAREST,
        src: data,
        width: width,
        height: 1,
        internalFormat: gl.RGBA32F,
        format: gl.RGBA,
        type: gl.FLOAT,
    });
}

function detectModelType(models, channel_n = 12) {
    // Detect if model is RD, NCA, rotinv_nca, or sqzast_nca based on layer 0 input dimensions
    if (!models.layers || models.layers.length < 1) {
        console.warn('No layers found in model, defaulting to NCA. models object:', models);
        return 'nca';
    }

    // Check if model_type is explicitly specified in JSON
    if (models.model_type) {
        console.log(`Model type explicitly set to: ${models.model_type}`);
        if (models.model_type === 'sqzast_nca') {
            console.log('  SQZAST NCA: theta channel has fixed dynamics (not learned)');
        }
        return models.model_type;
    }

    const layer0 = models.layers[0];
    const layer0_shape = layer0.shape;  // [input_dim, output_dim]

    // Subtract 1 from input_dim if bias is present (bias row at end)
    const input_dim = layer0.bias ? layer0_shape[0] - 1 : layer0_shape[0];

    // RD models: w1 takes raw state directly (channel_n inputs, no perception)
    // NCA models: w1 takes perception vector (4 * channel_n inputs)
    if (input_dim === channel_n) {
        console.log(`Detected RD model: input_dim=${input_dim} = ${channel_n} (state-only, no perception)`);
        return 'rd';
    } else if (input_dim === 4 * channel_n) {
        console.log(`Detected NCA model: input_dim=${input_dim} = 4 × ${channel_n} (identity + sobel_x + sobel_y + laplacian)`);
        return 'nca';
    } else {
        console.warn(`Unknown model type with input_dim=${input_dim}, channel_n=${channel_n}, defaulting to NCA`);
        return 'nca';
    }
}

export class NoiseNCA {
    constructor(gl, models, gridSize, gui) {
        // models is basically the json file

        self = this;
        this.gl = gl;

        // Detect model type (RD or NCA) before setting up anything else
        const lastLayer = models.layers[models.layers.length - 1];
        // Use channel_n from JSON if available (actual channel count before WebGL padding)
        // Otherwise fall back to lastLayer dimensions (may be padded to multiple of 4)
        const channel_n = models.channel_n !== undefined ? models.channel_n : (lastLayer.out_n || lastLayer.shape[1]);
        this.channel_n = channel_n;  // Store actual channel count for perception shader
        this.modelType = detectModelType(models, channel_n);
        this.isRD = (this.modelType === 'rd');
        this.isRotInv = (this.modelType === 'rotinv_nca');
        this.isSqzast = (this.modelType === 'sqzast_nca');
        this.thetaChannel = models.theta_channel !== undefined ? models.theta_channel : 3;

        // alert(this.n_perception_scales)

        this.gridSize = gridSize || [96, 96];

        this.updateProbability = 0.5;
        this.shuffledMode = false; // changed
        // alert(this.shuffledMode)
        this.circular_padding = true;

        this.rotationAngle = 0.0;
        // Use dt from model JSON if present, otherwise default to 1.0
        this.dt = models.dt !== undefined ? models.dt : 1.0;
        this.dx = 1.0;
        this.dy = 1.0;
        this.epsilon = 0.0;
        this.alignment = 0;
        this.fuzz = 30.0;
        this.perceptionCircle = 0.0;
        this.arrowsCoef = 0.0;
        this.visMode = 'color';
        this.hexGrid = false;
        this.viewChannel = -1.0;  // -1 = RGB, -2 = grayscale, 0+ = hidden channel index
        this.invertColors = false;
        this.autoScaleVis = false;  // Auto-scale visualization based on actual min/max
        this.visScaleOffset = { scale: 1.0, offset: 0.5 };  // Computed scale/offset for visualization

        this.noise_level = models.noise_level;

        // Store RD-specific hyperparameters from training
        // Default diffusion coefficients: [0.125, 0.25, 0.5, 1.0] repeated for 12 channels
        this.diff_coef = models.diff_coef || [0.125, 0.25, 0.5, 1.0, 0.125, 0.25, 0.5, 1.0, 0.125, 0.25, 0.5, 1.0];
        this.reaction_rate = models.reaction_rate !== undefined ? models.reaction_rate : 1.0;
        this.diffusion_rate = models.diffusion_rate !== undefined ? models.diffusion_rate : 1.0;
        this.seed_type = models.seed_type || 'gaussian_blobs';

        // Growing NCA mode: uses normalized Sobel filters and stochastic updates
        // Detected from model JSON or can be set manually
        this.growingMode = models.growing_mode || false;
        this.fireRate = models.fire_rate !== undefined ? models.fire_rate : 0.5;

        // Growing NCA specific features
        this.useLifeMask = models.use_life_mask || false;  // Alpha-based life mask
        this.zeroPadding = models.zero_padding || false;   // Zero padding vs circular
        this.noiseAffectsAlpha = models.noise_affects_alpha || false;  // Whether noise affects alpha channel

        // Override circular_padding if zeroPadding is enabled
        if (this.zeroPadding) {
            this.circular_padding = false;
        }

        // Log hyperparameters if RD model
        if (this.isRD) {
            console.log('RD Model hyperparameters from JSON:');
            console.log('  dt:', this.dt);
            console.log('  noise_level:', this.noise_level);
            console.log('  diff_coef:', this.diff_coef);
            console.log('  reaction_rate:', this.reaction_rate);
            console.log('  diffusion_rate:', this.diffusion_rate);
            console.log('  seed_type:', this.seed_type);
        }

        // Log if Growing NCA mode
        if (this.growingMode) {
            console.log('Growing NCA mode enabled:');
            console.log('  growingMode:', this.growingMode);
            console.log('  fireRate:', this.fireRate);
            console.log('  useLifeMask:', this.useLifeMask);
            console.log('  zeroPadding:', this.zeroPadding);
            console.log('  noiseAffectsAlpha:', this.noiseAffectsAlpha);
        }

        // Log channel count info
        console.log('Model channel count:', this.channel_n);
        console.log('  channel_n from JSON:', models.channel_n);
        console.log('  depth4 (ceil(channel_n/4)):', Math.ceil(this.channel_n / 4));
        console.log('  perception_n (4 * channel_n):', 4 * this.channel_n);
        if (this.isRotInv) {
            console.log('Rotation-invariant NCA model:');
            console.log('  theta_channel:', this.thetaChannel);
        }
        if (this.isSqzast) {
            console.log('SQZAST NCA model (fixed theta dynamics):');
            console.log('  theta_channel:', this.thetaChannel);
            console.log('  Theta update rule: stochastic quenched-zoning active spin texture');
        }

        this.layers = [];
        this.altLayers = [];  // Alternate texture weights for blending
        this.blendFactor = 0.0;  // 0 = primary only, 1 = alt only, 0.5 = equal blend

        // Weight perturbation system
        this.originalModels = null;        // Store original unperturbed JSON
        this.perturbationVectors = [];     // Random noise for each layer
        this.weightPerturbation = 0.0;     // Perturbation magnitude (0.0-1.0)

        // Weight pruning system
        this.pruningMask = [];              // Uint8Array for each layer (1=prune, 0=keep)
        this.pruningIndices = [];           // Sorted global indices for debugging
        this.weightPruningFraction = 0.0;   // Pruning fraction (0.0-1.0)

        // Neuron pruning system
        this.neuronPruningMask = [];            // Uint8Array for each layer
        this.neuronPruningFraction = 0.0;       // Neuron pruning fraction (0.0-1.0)
        this.prunedNeuronIndices = [];          // Indices of pruned neurons for debugging

        // Channel dropout system
        this.channelDropoutMask = null;         // Float32Array, length = channelCount, 1.0=active, 0.0=dropped
        this.dropoutMaskTex = null;             // Texture containing dropout mask

        // RD diffusion coefficient texture
        this.diffCoefTex = null;                // Texture containing per-channel diffusion coefficients

        this.setWeights(models);

        const defs = (this.circular_padding ? '#define CIRCULAR_PADDING\n' : '') +
            (this.shuffledMode ? '#define SPARSE_UPDATE\n' : '') +
            (this.useLifeMask ? '#define USE_LIFE_MASK\n' : '');


        this.progs = createPrograms(gl, defs);

        // representing vertices of a square with two triangles
        this.quad = twgl.createBufferInfoFromArrays(gl, {
            position: [-1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0],
        });

        this.setupBuffers();
        const visNames = Object.getOwnPropertyNames(this.buf);
        visNames.push('color');

        if (gui) {
            gui.add(this, 'rotationAngle').min(0.0).max(360.0);
            gui.add(this, 'alignment', {cartesian: 0, polar: 1, bipolar: 2}).listen();
            //gui.add(this, 'fuzz').min(0.0).max(128.0);
            //gui.add(this, 'perceptionCircle').min(0.0).max(1.0);
            //gui.add(this, 'visMode', visNames);
            gui.add(this, 'hexGrid');

            gui.add(this, 'benchmark');

            // this.benchmark = ()=>{
            //   document.getElementById('log').insertAdjacentHTML('afterbegin', this.benchmark());
            // }


        }

        // Use appropriate initialization based on model type
        if (this.isRD) {
            // RD models need gaussian blob initialization with zero hidden channels
            this.initRDStyle(0.01, 3.0);
        } else {
            // NCA models use uniform random noise
            this.clearCircle(0, 0, 10000);
        }
    }

    setupBuffers() {
        const gl = this.gl;
        const [gridW, gridH] = this.gridSize;
        const shuffleH = Math.ceil(gridH * this.updateProbability);
        const shuffleCellN = shuffleH * gridW;
        const totalCellN = gridW * gridH;
        // const shuffleBuf = new Uint8Array(shuffleCellN * 4); // Indices of the cells to be updated
        // const unshuffleBuf = new Uint8Array(totalCellN * 4);

        const shuffleBuf = new Float32Array(shuffleCellN * 4); // Indices of the cells to be updated
        const unshuffleBuf = new Float32Array(totalCellN * 4);

        let k = 0;


        for (let i = 0; i < totalCellN; ++i) {
            // This exactly updates shuffleCellN of cells.
            if (Math.random() < (shuffleCellN - k) / (totalCellN - i)) {
                shuffleBuf[k * 4 + 0] = i % gridW;
                shuffleBuf[k * 4 + 1] = Math.floor(i / gridW);
                unshuffleBuf[i * 4 + 0] = k % gridW;
                unshuffleBuf[i * 4 + 1] = Math.floor(k / gridW);
                unshuffleBuf[i * 4 + 2] = 255;
                k += 1;
            }
        }
        this.shuffleTex = twgl.createTexture(gl, {
            minMag: gl.NEAREST, width: gridW, height: shuffleH, src: shuffleBuf,
            flipY: false, premultiplyAlpha: false,
            format: gl.RGBA,
            internalFormat: gl.RGBA32F,
            type: gl.FLOAT,
        });
        this.unshuffleTex = twgl.createTexture(gl, {
            minMag: gl.NEAREST,
            width: gridW,
            height: gridH,
            src: unshuffleBuf,
            flipY: false, premultiplyAlpha: false,
            format: gl.RGBA,
            internalFormat: gl.RGBA32F,
            type: gl.FLOAT,
        });
        this.shuffleOfs = [0, 0];

        const updateH = this.shuffledMode ? shuffleH : gridH;
        const perception_n = this.layers[0].in_n;
        const lastLayer = this.layers[this.layers.length - 1];
        // Use this.channel_n (actual count from JSON) instead of lastLayer.out_n (may be padded)
        const channel_n = this.channel_n;
        this.channelCount = channel_n;  // Store for external access

        // Initialize channel dropout mask (all active by default)
        this.channelDropoutMask = new Float32Array(channel_n).fill(1.0);

        const stateQuantization = lastLayer.quantScaleZero;
        const no_quantization = [1.0, 0.0];
        this.buf = {
            control: createTensor(gl, gridW, gridH, 4, [255.0, 0.0], false),
            state: createTensor(gl, gridW, gridH, channel_n, no_quantization, true),
            newState: createTensor(gl, gridW, gridH, channel_n, no_quantization, true),
            perception0: createTensor(gl, gridW, updateH, perception_n, no_quantization, true),

        };

        // Create dropout mask texture
        this.dropoutMaskTex = createDropoutMaskTexture(gl, channel_n, this.channelDropoutMask);

        // Create diffusion coefficient texture for RD models
        // Ensure we have exactly channel_n coefficients
        let diffCoefs = this.diff_coef.slice(0, channel_n);
        // Pad if needed
        while (diffCoefs.length < channel_n) {
            diffCoefs.push(this.diff_coef[diffCoefs.length % this.diff_coef.length]);
        }
        this.diffCoefTex = createDiffCoefTexture(gl, diffCoefs);

        // For now we only support multi-scale perception with 2 scales


        for (let i = 0; i < this.layers.length; ++i) {
            const layer = this.layers[i];
            // this.buf[`layer${i}`] = createTensor(gl, gridW, updateH, layer.out_n, layer.quantScaleZero, false);
            this.buf[`layer${i}`] = createTensor(gl, gridW, updateH, layer.out_n, no_quantization, true);
            // Alternate texture layer buffers
            this.buf[`altLayer${i}`] = createTensor(gl, gridW, updateH, layer.out_n, no_quantization, true);
        }
        // Buffer for alternate texture's final update
        this.buf.altUpdate = createTensor(gl, gridW, updateH, channel_n, no_quantization, true);
    }

    step(stage) {
        stage = stage || 'all';
        if (!this.layers.every(l => l.ready))
            return;

        const seed = Math.random() * 1000;

        if (stage == 'all') {
            const [gridW, gridH] = this.gridSize;
            // Random point on the grid
            this.shuffleOfs = [Math.floor(Math.random() * gridW), Math.floor(Math.random() * gridH)];
        }

        if (stage == 'all' || stage == 'Perception') {
            if (this.isRD) {
                // For RD models: skip perception, use state directly as input to FC layers
                // The laplacian will be computed in the update shader
                this.runLayer(self.progs.perception, this.buf.perception0, {
                    u_input: this.buf.state, u_angle: this.rotationAngle / 180.0 * Math.PI,
                    u_alignment: this.alignment, u_hexGrid: this.hexGrid, u_dx: this.dx, u_dy: this.dy,
                    u_seed: seed, u_updateProbability: this.updateProbability,
                    u_isRD: 0.0,  // Use identity-only perception (filterBand 0)
                    u_rdStateOnly: 1.0,  // Signal to only copy state, no filters
                });
            } else {
                // For rotinv and sqzast models, use theta-based gradient rotation
                const usesRotatedGradients = this.isRotInv || this.isSqzast;
                this.runLayer(self.progs.perception, this.buf.perception0, {
                    u_input: this.buf.state, u_angle: this.rotationAngle / 180.0 * Math.PI,
                    u_alignment: this.alignment, u_hexGrid: this.hexGrid, u_dx: this.dx, u_dy: this.dy,
                    u_seed: seed, u_updateProbability: this.updateProbability,
                    u_isRD: 0.0,
                    u_growingMode: this.growingMode ? 1.0 : 0.0,
                    u_channelCount: this.channel_n,
                    u_isRotInv: usesRotatedGradients ? 1.0 : 0.0,
                    u_thetaChannel: this.thetaChannel,
                });
            }
        }


        // For RD models, use state buffer directly; for NCA, use perception
        let inputBuf = this.isRD ? this.buf.state : this.buf.perception0;

        // Compute primary texture update
        for (let i = 0; i < this.layers.length; ++i) {
            if (stage == 'all' || stage == `FC Layer${i + 1}`)
                var relu = i == 0 ? true : false;
            this.runDense(this.buf[`layer${i}`], inputBuf, this.layers[i], relu, seed);
            inputBuf = this.buf[`layer${i}`];
        }
        
        // Compute alternate texture update if we have alt weights and blending
        const hasAltLayers = this.altLayers && this.altLayers.length > 0 && this.altLayers.every(l => l.ready);
        // For RD models, use state buffer directly; for NCA, use perception
        let altInputBuf = this.isRD ? this.buf.state : this.buf.perception0;
        
        if (hasAltLayers && this.blendFactor > 0) {
            for (let i = 0; i < this.altLayers.length; ++i) {
                if (stage == 'all' || stage == `FC Layer${i + 1}`)
                    var relu = i == 0 ? true : false;
                this.runDense(this.buf[`altLayer${i}`], altInputBuf, this.altLayers[i], relu, seed);
                altInputBuf = this.buf[`altLayer${i}`];
            }
        }
        
        if (stage == 'all' || stage == 'Update') {
            if (this.isRD && hasAltLayers && this.blendFactor > 0) {
                // Use blended RD update shader (blend two reaction terms with shared diffusion)
                this.runLayer(this.progs.blendedRdUpdate, this.buf.newState, {
                    u_input: this.buf.state, u_update: inputBuf, u_altUpdate: altInputBuf,
                    u_unshuffleTex: this.unshuffleTex, u_dt: this.dt, u_epsilon: this.epsilon,
                    u_seed: seed, u_updateProbability: this.updateProbability,
                    u_blendFactor: this.blendFactor,
                    u_dropoutMask: this.dropoutMaskTex,
                    u_dropoutMask_size: [Math.ceil(this.channelCount / 4), 1],
                    u_diffCoefTex: this.diffCoefTex,
                    u_diffCoefTex_size: [Math.ceil(this.channelCount / 4), 1],
                    u_reactionRate: this.reaction_rate,
                    u_diffusionRate: this.diffusion_rate
                });
            } else if (this.isRD) {
                // Use RD-specific update shader that computes laplacian diffusion term
                this.runLayer(this.progs.rdUpdate, this.buf.newState, {
                    u_input: this.buf.state, u_update: inputBuf,
                    u_unshuffleTex: this.unshuffleTex, u_dt: this.dt, u_epsilon: this.epsilon,
                    u_seed: seed, u_updateProbability: this.updateProbability,
                    u_dropoutMask: this.dropoutMaskTex,
                    u_dropoutMask_size: [Math.ceil(this.channelCount / 4), 1],
                    u_diffCoefTex: this.diffCoefTex,
                    u_diffCoefTex_size: [Math.ceil(this.channelCount / 4), 1],
                    u_reactionRate: this.reaction_rate,
                    u_diffusionRate: this.diffusion_rate
                });
            } else if (hasAltLayers && this.blendFactor > 0) {
                // Use blended update shader for NCA
                this.runLayer(this.progs.blendedUpdate, this.buf.newState, {
                    u_input: this.buf.state, u_update: inputBuf, u_altUpdate: altInputBuf,
                    u_unshuffleTex: this.unshuffleTex, u_dt: this.dt, u_epsilon: this.epsilon,
                    u_seed: seed, u_updateProbability: this.updateProbability,
                    u_blendFactor: this.blendFactor,
                    u_dropoutMask: this.dropoutMaskTex,
                    u_dropoutMask_size: [Math.ceil(this.channelCount / 4), 1],
                    u_growingMode: this.growingMode ? 1.0 : 0.0,
                    u_fireRate: this.fireRate,
                    u_noiseAffectsAlpha: this.noiseAffectsAlpha ? 1.0 : 0.0,
                });
            } else {
                // Use regular update shader (no blending)
                this.runLayer(this.progs.update, this.buf.newState, {
                    u_input: this.buf.state, u_update: inputBuf,
                    u_unshuffleTex: this.unshuffleTex, u_dt: this.dt, u_epsilon: this.epsilon,
                    u_seed: seed, u_updateProbability: this.updateProbability,
                    u_dropoutMask: this.dropoutMaskTex,
                    u_dropoutMask_size: [Math.ceil(this.channelCount / 4), 1],
                    u_growingMode: this.growingMode ? 1.0 : 0.0,
                    u_fireRate: this.fireRate,
                    u_noiseAffectsAlpha: this.noiseAffectsAlpha ? 1.0 : 0.0,
                });
            }

            // SQZAST: Apply fixed theta dynamics AFTER the learned update
            // This replaces the learned theta update with the SQZAST rule
            if (this.isSqzast) {
                // newState currently has the learned update applied to all channels
                // We need to override the theta channel with SQZAST dynamics
                // sqzastThetaUpdate reads:
                //   - u_input (original state) for theta and neighbor theta values
                //   - u_newState (post-learned) for all other channels
                // Outputs combined state where theta uses SQZAST, others use learned
                // Write to altUpdate as temp buffer, then copy to newState
                this.runLayer(this.progs.sqzastThetaUpdate, this.buf.altUpdate, {
                    u_input: this.buf.state,  // Original state (before this step's update)
                    u_newState: this.buf.newState,  // Post-learned-update state
                    u_seed: seed,
                    u_dt: this.dt,
                    u_thetaChannel: this.thetaChannel,
                });
                // altUpdate now has combined result, swap it into newState
                [this.buf.newState, this.buf.altUpdate] = [this.buf.altUpdate, this.buf.newState];
            }
        }

        if (stage == 'all') {
            [this.buf.state, this.buf.newState] = [this.buf.newState, this.buf.state];
        }
    }

    updateChannelDropout(channelIndex, isActive) {
        // Update mask array
        this.channelDropoutMask[channelIndex] = isActive ? 1.0 : 0.0;

        // Recreate texture
        const gl = this.gl;
        if (this.dropoutMaskTex) {
            gl.deleteTexture(this.dropoutMaskTex);
        }
        this.dropoutMaskTex = createDropoutMaskTexture(gl, this.channelCount, this.channelDropoutMask);

        console.log(`Channel ${channelIndex} ${isActive ? 'enabled' : 'dropped out'}`);
    }

    benchmark() {
        const gl = this.gl;
        // const flushBuf = new Uint8Array(4);
        const flushBuf = new Float32Array(4);
        const flush = buf => {
            buf = buf || this.buf.state;
            // gl.flush/finish don't seem to do anything, so reading a single
            // pixel from the state buffer to flush the GPU command pipeline
            twgl.bindFramebufferInfo(gl, buf.fbi);
            gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, flushBuf);
        }

        // const flushBuf = new Uint8Array(4);
        // const flush = buf=>{
        //     buf = buf || this.buf.state;
        //     // gl.flush/finish don't seem to do anything, so reading a single
        //     // pixel from the state buffer to flush the GPU command pipeline
        //     twgl.bindFramebufferInfo(gl, buf.fbi);
        //     gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, flushBuf);
        // }

        flush();
        const stepN = 500;
        const start = Date.now();
        for (let i = 0; i < stepN; ++i)
            this.step();
        flush();
        const total = (Date.now() - start) / stepN;

        const ops = [];
        if (this.n_perception_scales > 1) {
            ops.push('Multi-Scale Perception');
        } else {
            ops.push('Perception')
        }


        for (let i = 0; i < this.layers.length; ++i)
            ops.push(`FC Layer${i + 1}`);
        ops.push('Update');
        let perOpTotal = 0.0;
        const perOp = [];
        for (const op of ops) {
            const start = Date.now();
            for (let i = 0; i < stepN; ++i) {
                this.step(op);
            }
            flush(this.buf[op]);
            const dt = (Date.now() - start) / stepN;
            perOpTotal += dt
            perOp.push([op, dt]);
        }
        const perOpStr = perOp.map((p) => {
            const [programName, dt] = p;
            const percent = 100.0 * dt / perOpTotal;
            return `${programName}: ${percent.toFixed(1)}%`;
        }).join('\n');
        const T = this.n_perception_scales > 1 ? 64.0 : 24.0;
        var result = `FPS: ${(1000.0 / (total * T)).toFixed(2)}, ${(total).toFixed(2)} ms/step, ${(1000.0 / total).toFixed(2)} steps/sec\n` + perOpStr + '\n\n'
        // document.getElementById('log').innerHTML = `${(total).toFixed(2)} ms/step, ${(1000.0 / total).toFixed(2)} step/sec\n` + perOpStr + '\n\n'
        alert(result);
    }

    paint(x, y, r, brush) {
        // the model idx is passed as the brush
        this.runLayer(this.progs.paint, this.buf.control, {
            u_pos: [x, y], u_r: r, u_brush: [brush, 0, 0, 0], u_hexGrid: this.hexGrid, u_zoom: 1.0, u_control: true,
        });
    }

    clearCircle(x, y, r, brush, zoom = 1.0, noiseLevel = 1.0) {
        let u_seed = Math.random() * 1000.0;
        self.runLayer(self.progs.paint, this.buf.state, {
            u_pos: [x, y],
            u_r: r,
            u_brush: [0.0, 0.0, 0.0, 0.0],
            u_hexGrid: this.hexGrid,
            u_zoom: zoom,
            u_seed: u_seed,
            u_control: false,
            u_noise_level: noiseLevel,
            u_noiseAffectsAlpha: this.noiseAffectsAlpha ? 1.0 : 0.0,
        });
    }

    // RD-style initialization: gaussian blobs in RGB, zeros in hidden channels
    initRDStyle(spotProb = 0.01, spread = 3.0) {
        const gl = this.gl;
        const [w, h] = this.gridSize;
        const channelCount = this.channelCount;

        // Create gaussian blob pattern on CPU
        // Step 1: Create sparse random spots
        const spots = new Float32Array(w * h);
        for (let i = 0; i < w * h; i++) {
            spots[i] = Math.random() < spotProb ? 1.0 : 0.0;
        }

        // Step 2: Apply Gaussian blur (simple box blur approximation, repeated)
        const blurred = new Float32Array(w * h);
        const temp = new Float32Array(w * h);

        // Copy spots to blurred
        for (let i = 0; i < spots.length; i++) {
            blurred[i] = spots[i];
        }

        // Apply blur passes (approximate gaussian with repeated box blur)
        const blurPasses = Math.ceil(spread);
        for (let pass = 0; pass < blurPasses; pass++) {
            // Horizontal pass
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    let sum = 0;
                    for (let dx = -1; dx <= 1; dx++) {
                        const nx = (x + dx + w) % w;  // Circular padding
                        sum += blurred[y * w + nx];
                    }
                    temp[y * w + x] = sum / 3.0;
                }
            }
            // Vertical pass
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    let sum = 0;
                    for (let dy = -1; dy <= 1; dy++) {
                        const ny = (y + dy + h) % h;  // Circular padding
                        sum += temp[ny * w + x];
                    }
                    blurred[y * w + x] = sum / 3.0;
                }
            }
        }

        // Scale by spread^2 to compensate for blur normalization
        const scale = spread * spread;
        for (let i = 0; i < blurred.length; i++) {
            blurred[i] *= scale;
        }

        // Step 3: Create state tensor data
        // State is stored as packed RGBA textures in a grid
        const depth4 = Math.ceil(channelCount / 4);
        const gridW = Math.ceil(Math.sqrt(depth4));
        const gridH = Math.floor((depth4 + gridW - 1) / gridW);
        const texW = w * gridW;
        const texH = h * gridH;

        const stateData = new Float32Array(texW * texH * 4);

        // Fill the state data
        for (let ch4 = 0; ch4 < depth4; ch4++) {
            const tileX = ch4 % gridW;
            const tileY = Math.floor(ch4 / gridW);

            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const texX = tileX * w + x;
                    const texY = tileY * h + y;
                    const texIdx = (texY * texW + texX) * 4;

                    const spatialIdx = y * w + x;
                    const blobVal = blurred[spatialIdx];

                    // For each of the 4 channels in this pack
                    for (let c = 0; c < 4; c++) {
                        const channelIdx = ch4 * 4 + c;
                        if (channelIdx < channelCount) {
                            if (channelIdx < 3) {
                                // RGB channels: gaussian blobs, centered around 0 (display adds 0.5)
                                stateData[texIdx + c] = blobVal - 0.5;
                            } else {
                                // Hidden channels: exactly 0
                                stateData[texIdx + c] = 0.0;
                            }
                        }
                    }
                }
            }
        }

        // Upload to GPU
        twgl.setTextureFromArray(gl, this.buf.state.tex, stateData, {
            width: texW,
            height: texH,
            format: gl.RGBA,
            internalFormat: gl.RGBA32F,
            type: gl.FLOAT,
            minMag: gl.NEAREST,
        });

        console.log(`Initialized RD-style: ${w}x${h}, spotProb=${spotProb}, spread=${spread}`);
    }

    // Growing NCA-style initialization: zeros everywhere, hidden channels activated at center
    // This matches the seed from nca_experiments.ipynb / Growing NCA paper
    initCenterSeed() {
        const gl = this.gl;
        const [w, h] = this.gridSize;
        const channelCount = this.channelCount;

        // State is stored as packed RGBA textures in a grid
        const depth4 = Math.ceil(channelCount / 4);
        const gridW = Math.ceil(Math.sqrt(depth4));
        const gridH = Math.floor((depth4 + gridW - 1) / gridW);
        const texW = w * gridW;
        const texH = h * gridH;

        const stateData = new Float32Array(texW * texH * 4);

        // Center coordinates
        const centerX = Math.floor(w / 2);
        const centerY = Math.floor(h / 2);

        // Fill the state data
        for (let ch4 = 0; ch4 < depth4; ch4++) {
            const tileX = ch4 % gridW;
            const tileY = Math.floor(ch4 / gridW);

            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const texX = tileX * w + x;
                    const texY = tileY * h + y;
                    const texIdx = (texY * texW + texX) * 4;

                    const isCenter = (x === centerX && y === centerY);

                    // For each of the 4 channels in this pack
                    for (let c = 0; c < 4; c++) {
                        const channelIdx = ch4 * 4 + c;
                        if (channelIdx < channelCount) {
                            if (channelIdx < 3) {
                                // RGB channels: always 0 (will display as 0.5 gray)
                                stateData[texIdx + c] = 0.0;
                            } else {
                                // Hidden channels (3+): 1.0 at center, 0.0 elsewhere
                                stateData[texIdx + c] = isCenter ? 1.0 : 0.0;
                            }
                        }
                    }
                }
            }
        }

        // Upload to GPU
        twgl.setTextureFromArray(gl, this.buf.state.tex, stateData, {
            width: texW,
            height: texH,
            format: gl.RGBA,
            internalFormat: gl.RGBA32F,
            type: gl.FLOAT,
            minMag: gl.NEAREST,
        });

        console.log(`Initialized center seed: ${w}x${h}, center=(${centerX}, ${centerY})`);
    }

    // Manually set RD mode (for models that aren't auto-detected)
    setRDMode(isRD) {
        this.isRD = isRD;
        console.log(`RD Mode ${isRD ? 'enabled' : 'disabled'}`);
    }

    // Compute min/max of RGB channels for auto-scaling visualization
    // Returns { min, max } for channels 0-2 (RGB) or specific channel if viewing hidden
    computeVisMinMax() {
        const gl = this.gl;
        const buf = this.buf.state;

        twgl.bindFramebufferInfo(gl, buf.fbi);
        const data = new Float32Array(buf.fbi.width * buf.fbi.height * 4);
        gl.readPixels(0, 0, buf.fbi.width, buf.fbi.height, gl.RGBA, gl.FLOAT, data);

        // Unbind framebuffer and restore viewport to canvas size
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

        const [w, h] = this.gridSize;
        const depth4 = Math.ceil(this.channelCount / 4);
        const gridW = Math.ceil(Math.sqrt(depth4));

        let min = Infinity, max = -Infinity;
        let sum = 0, count = 0;

        // Determine which channels to analyze
        let channelsToCheck;
        if (this.viewChannel >= 0) {
            // Viewing a specific hidden channel
            channelsToCheck = [Math.floor(this.viewChannel)];
        } else {
            // Viewing RGB (channels 0, 1, 2)
            channelsToCheck = [0, 1, 2];
        }

        // For each channel we want to check
        for (const ch of channelsToCheck) {
            const ch4 = Math.floor(ch / 4);
            const chOffset = ch % 4;
            const tileX = ch4 % gridW;
            const tileY = Math.floor(ch4 / gridW);

            // Iterate over all pixels in this channel's tile
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const texX = tileX * w + x;
                    const texY = tileY * h + y;
                    const texIdx = (texY * buf.fbi.width + texX) * 4 + chOffset;
                    const val = data[texIdx];

                    if (!isNaN(val) && isFinite(val)) {
                        min = Math.min(min, val);
                        max = Math.max(max, val);
                        sum += val;
                        count++;
                    }
                }
            }
        }

        const mean = count > 0 ? sum / count : 0;
        return { min, max, mean };
    }

    // Update scale/offset for auto-scaling visualization
    updateVisScaleOffset() {
        if (!this.autoScaleVis) {
            // Reset to default (for non-growing: offset=0.5, scale=1.0)
            if (this.growingMode) {
                this.visScaleOffset = { scale: 1.0, offset: 0.0 };
            } else {
                this.visScaleOffset = { scale: 1.0, offset: 0.5 };
            }
            return;
        }

        const { min, max } = this.computeVisMinMax();

        if (min === Infinity || max === -Infinity || min === max) {
            // No valid data or all same value
            this.visScaleOffset = { scale: 1.0, offset: this.growingMode ? 0.0 : 0.5 };
            return;
        }

        // We want to map [min, max] -> [0, 1]
        // output = (input - min) / (max - min)
        // output = input * scale + offset
        // scale = 1 / (max - min)
        // offset = -min / (max - min)
        const range = max - min;
        const scale = 1.0 / range;
        const offset = -min / range;

        this.visScaleOffset = { scale, offset };
    }

    // Debug: read buffer values and print stats
    debugReadBuffer(bufName = 'state') {
        const gl = this.gl;
        const buf = this.buf[bufName];
        if (!buf) {
            console.error(`Buffer '${bufName}' not found`);
            return;
        }
        twgl.bindFramebufferInfo(gl, buf.fbi);
        const data = new Float32Array(buf.fbi.width * buf.fbi.height * 4);
        gl.readPixels(0, 0, buf.fbi.width, buf.fbi.height, gl.RGBA, gl.FLOAT, data);

        let min = Infinity, max = -Infinity, sum = 0, nanCount = 0;
        for (let i = 0; i < data.length; i++) {
            if (isNaN(data[i])) {
                nanCount++;
            } else {
                min = Math.min(min, data[i]);
                max = Math.max(max, data[i]);
                sum += data[i];
            }
        }
        const mean = sum / (data.length - nanCount);
        console.log(`Buffer '${bufName}': min=${min.toFixed(4)}, max=${max.toFixed(4)}, mean=${mean.toFixed(4)}, NaN count=${nanCount}`);
        return { min, max, mean, nanCount, data };
    }

    // Debug: run one step and print all buffer states
    debugStep() {
        console.log("=== Before step ===");
        this.debugReadBuffer('state');

        console.log("=== Running step ===");
        console.log("isRD:", this.isRD);
        console.log("Layer dimensions:", this.layers.map((l, i) => `L${i}: ${l.in_n}->${l.out_n}`).join(", "));

        // Run perception
        const seed = Math.random() * 1000;
        if (this.isRD) {
            console.log("Skipping perception (RD mode - using state directly)");
        } else {
            const usesRotatedGradients = this.isRotInv || this.isSqzast;
            this.runLayer(self.progs.perception, this.buf.perception0, {
                u_input: this.buf.state, u_angle: this.rotationAngle / 180.0 * Math.PI,
                u_alignment: this.alignment, u_hexGrid: this.hexGrid, u_dx: this.dx, u_dy: this.dy,
                u_seed: seed, u_updateProbability: this.updateProbability,
                u_isRD: 0.0,
                u_channelCount: this.channel_n,
                u_growingMode: this.growingMode ? 1.0 : 0.0,
                u_isRotInv: usesRotatedGradients ? 1.0 : 0.0,
                u_thetaChannel: this.thetaChannel,
            });
            console.log("After perception:");
            this.debugReadBuffer('perception0');
        }

        // Run FC layers
        let inputBuf = this.isRD ? this.buf.state : this.buf.perception0;
        for (let i = 0; i < this.layers.length; i++) {
            const relu = i === 0;
            this.runDense(this.buf[`layer${i}`], inputBuf, this.layers[i], relu, seed);
            console.log(`After layer${i} (relu=${relu}):`);
            this.debugReadBuffer(`layer${i}`);
            inputBuf = this.buf[`layer${i}`];
        }

        // Run update
        if (this.isRD) {
            console.log("Running RD update (with diffusion)...");
            console.log(`  dt=${this.dt}, reaction_rate=${this.reaction_rate}, diffusion_rate=${this.diffusion_rate}`);
            this.runLayer(this.progs.rdUpdate, this.buf.newState, {
                u_input: this.buf.state, u_update: inputBuf,
                u_unshuffleTex: this.unshuffleTex, u_dt: this.dt, u_epsilon: this.epsilon,
                u_seed: seed, u_updateProbability: this.updateProbability,
                u_dropoutMask: this.dropoutMaskTex,
                u_dropoutMask_size: [Math.ceil(this.channelCount / 4), 1],
                u_diffCoefTex: this.diffCoefTex,
                u_diffCoefTex_size: [Math.ceil(this.channelCount / 4), 1],
                u_reactionRate: this.reaction_rate,
                u_diffusionRate: this.diffusion_rate
            });
        } else {
            console.log("Running standard update...");
            this.runLayer(this.progs.update, this.buf.newState, {
                u_input: this.buf.state, u_update: inputBuf,
                u_unshuffleTex: this.unshuffleTex, u_dt: this.dt, u_epsilon: this.epsilon,
                u_seed: seed, u_updateProbability: this.updateProbability,
                u_dropoutMask: this.dropoutMaskTex,
                u_dropoutMask_size: [Math.ceil(this.channelCount / 4), 1]
            });
        }

        console.log("After update:");
        this.debugReadBuffer('newState');

        // Swap buffers
        [this.buf.state, this.buf.newState] = [this.buf.newState, this.buf.state];

        console.log("=== After step (final state) ===");
        this.debugReadBuffer('state');
    }

    setWeights(models) {
        const gl = this.gl;
        // Store original models and generate initial perturbation and pruning vectors
        this.originalModels = this.deepCloneModels(models);
        this.generatePerturbationVectors(models);
        this.generatePruningMask(models, this.weightPruningFraction);
        this.generateNeuronPruningMask(models, this.neuronPruningFraction);

        // Apply transformations in order: neuron pruning, weight pruning, then perturbations
        let modelsToUse = models;

        // Step 1: Apply neuron pruning if active
        if (this.neuronPruningFraction > 0) {
            modelsToUse = this.applyNeuronPruning(modelsToUse);
        }

        // Step 2: Apply weight pruning if active
        if (this.weightPruningFraction > 0) {
            modelsToUse = this.applyPruning(modelsToUse);
        }

        // Step 3: Apply perturbations if active
        if (this.weightPerturbation > 0) {
            modelsToUse = this.applyPerturbations(modelsToUse, this.weightPerturbation);
        }

        this.layers.forEach(layer => { if (layer.tex) gl.deleteTexture(layer.tex); });
        this.layers = modelsToUse.layers.map(layer => createDenseInfo(gl, layer));
    }

    deepCloneModels(models) {
        return {
            model_names: models.model_names,
            noise_level: models.noise_level,
            layers: models.layers.map(layer => ({
                ...layer,
                data_flatten: layer.data_flatten ? [...layer.data_flatten] : undefined,
                data: layer.data ? layer.data : undefined,
            }))
        };
    }

    generatePerturbationVectors(models) {
        this.perturbationVectors = [];
        this.perturbationScales = [];  // Store std dev for each layer

        for (let i = 0; i < models.layers.length; i++) {
            const layer = models.layers[i];
            if (layer.data_flatten) {
                const data = layer.data_flatten;
                const length = data.length;

                // For layer 0, the data is organized as [in_channels+1, fc_dim] flattened
                // The last in_channel is the bias (96 elements at the end for standard models)
                // For layer 1, there's no bias, just weights [fc_dim, out_channels]

                const perturbations = new Float32Array(length);

                if (i === 0 && layer.bias) {
                    // Layer 0 has bias in the last row
                    // layer.shape gives [in_channels+1, out_channels], e.g., [49, 96]
                    // data_flatten has this shape flattened row-major
                    const inChannels = layer.shape[0];  // e.g., 49 (includes bias row)
                    const outChannels = layer.shape[1];  // e.g., 96
                    const weightElements = (inChannels - 1) * outChannels;  // First 48 rows
                    const biasElements = outChannels;  // Last row

                    // Compute std dev for weights (first weightElements)
                    let weightSum = 0;
                    for (let j = 0; j < weightElements; j++) {
                        weightSum += data[j];
                    }
                    const weightMean = weightSum / weightElements;

                    let weightSumSq = 0;
                    for (let j = 0; j < weightElements; j++) {
                        const diff = data[j] - weightMean;
                        weightSumSq += diff * diff;
                    }
                    const weightStd = Math.sqrt(weightSumSq / weightElements);

                    // Compute std dev for biases (last biasElements)
                    let biasSum = 0;
                    for (let j = weightElements; j < length; j++) {
                        biasSum += data[j];
                    }
                    const biasMean = biasSum / biasElements;

                    let biasSumSq = 0;
                    for (let j = weightElements; j < length; j++) {
                        const diff = data[j] - biasMean;
                        biasSumSq += diff * diff;
                    }
                    const biasStd = Math.sqrt(biasSumSq / biasElements);

                    console.log(`Layer ${i}: weight std = ${weightStd.toFixed(6)} (${weightElements} params), bias std = ${biasStd.toFixed(6)} (${biasElements} params)`);

                    // Generate perturbations: weights scaled by weightStd, biases by biasStd
                    for (let j = 0; j < weightElements; j++) {
                        perturbations[j] = (Math.random() * 2 - 1) * weightStd;
                    }
                    for (let j = weightElements; j < length; j++) {
                        perturbations[j] = (Math.random() * 2 - 1) * biasStd;
                    }

                    this.perturbationScales.push({weight: weightStd, bias: biasStd});

                } else {
                    // Layer 1 or layers without bias: just weights
                    let sum = 0;
                    for (let j = 0; j < length; j++) {
                        sum += data[j];
                    }
                    const mean = sum / length;

                    let sumSq = 0;
                    for (let j = 0; j < length; j++) {
                        const diff = data[j] - mean;
                        sumSq += diff * diff;
                    }
                    const std = Math.sqrt(sumSq / length);

                    console.log(`Layer ${i}: weight std = ${std.toFixed(6)} (${length} params, no bias)`);

                    // Generate perturbations scaled by std dev
                    for (let j = 0; j < length; j++) {
                        perturbations[j] = (Math.random() * 2 - 1) * std;
                    }

                    this.perturbationScales.push({weight: std, bias: 0});
                }

                this.perturbationVectors.push(perturbations);
            } else {
                this.perturbationVectors.push(null);
                this.perturbationScales.push({weight: 0, bias: 0});
            }
        }
    }

    applyPerturbations(models, magnitude) {
        const perturbed = this.deepCloneModels(models);
        for (let i = 0; i < perturbed.layers.length; i++) {
            const layer = perturbed.layers[i];
            const perturbVector = this.perturbationVectors[i];
            if (layer.data_flatten && perturbVector) {
                for (let j = 0; j < layer.data_flatten.length; j++) {
                    layer.data_flatten[j] += perturbVector[j] * magnitude;
                }
            }
        }
        return perturbed;
    }

    updatePerturbationMagnitude(magnitude) {
        this.weightPerturbation = magnitude;
        if (this.originalModels) {
            const gl = this.gl;

            // Apply transformations in order: neuron pruning, weight pruning, then perturbations
            let modelsToUse = this.originalModels;

            if (this.neuronPruningFraction > 0) {
                modelsToUse = this.applyNeuronPruning(modelsToUse);
            }

            if (this.weightPruningFraction > 0) {
                modelsToUse = this.applyPruning(modelsToUse);
            }

            if (magnitude > 0) {
                modelsToUse = this.applyPerturbations(modelsToUse, magnitude);
            }

            this.layers.forEach(layer => { if (layer.tex) gl.deleteTexture(layer.tex); });
            this.layers = modelsToUse.layers.map(layer => createDenseInfo(gl, layer));
        }
    }

    resetPerturbations() {
        if (this.originalModels) {
            this.generatePerturbationVectors(this.originalModels);
            this.updatePerturbationMagnitude(this.weightPerturbation);
        }
    }

    generatePruningMask(models, fraction) {
        // Reset pruning data structures
        this.pruningMask = [];
        this.pruningIndices = [];

        if (fraction <= 0) {
            // No pruning - initialize empty masks
            for (let i = 0; i < models.layers.length; i++) {
                const layer = models.layers[i];
                if (layer.data_flatten) {
                    this.pruningMask.push(new Uint8Array(layer.data_flatten.length));
                } else {
                    this.pruningMask.push(null);
                }
            }
            return;
        }

        // Step 1: Collect all weights with their DEQUANTIZED absolute values and global indices
        const allWeights = [];
        let globalIndex = 0;

        for (let layerIdx = 0; layerIdx < models.layers.length; layerIdx++) {
            const layer = models.layers[layerIdx];
            if (!layer.data_flatten) continue;

            const data = layer.data_flatten;
            const length = data.length;
            const scale = layer.scale || 1.0;
            const center = layer.center || 0.0;

            // Determine weight range (exclude biases)
            let weightEnd = length;
            if (layerIdx === 0 && layer.bias) {
                // Layer 0: exclude last 96 elements (biases at end)
                const [inChannels, outChannels] = layer.shape;  // [49, 96]
                weightEnd = (inChannels - 1) * outChannels;     // 48 * 96 = 4608
            }

            // Collect weights with DEQUANTIZED absolute values
            for (let i = 0; i < weightEnd; i++) {
                const quantizedValue = data[i];
                const realValue = (quantizedValue - center) * scale;
                allWeights.push({
                    absValue: Math.abs(realValue),  // Use dequantized absolute value!
                    layerIdx: layerIdx,
                    localIdx: i,
                    globalIdx: globalIndex++
                });
            }

            // Skip biases in global indexing
            if (layerIdx === 0 && layer.bias) {
                globalIndex += (length - weightEnd);
            }
        }

        // Step 2: Sort by absolute value (ascending - smallest first)
        allWeights.sort((a, b) => a.absValue - b.absValue);

        // Step 3: Determine how many weights to prune
        const totalWeights = allWeights.length;
        const numToPrune = Math.round(fraction * totalWeights);

        console.log(`Pruning: ${numToPrune} / ${totalWeights} weights (${(fraction * 100).toFixed(1)}%)`);

        // Debug: Show smallest and largest pruned weights
        if (numToPrune > 0) {
            // Smallest pruned weight (most confident to prune)
            const smallest = allWeights[0];
            const smallestLayerName = smallest.layerIdx === 0 ? 'W1 (Layer 0)' : 'W2 (Layer 1)';
            const smallestLayer = models.layers[smallest.layerIdx];
            const smallestQuantized = smallestLayer.data_flatten[smallest.localIdx];
            const smallestDequantized = (smallestQuantized - smallestLayer.center) * smallestLayer.scale;

            // Largest pruned weight (least confident to prune - right at the threshold)
            const largest = allWeights[numToPrune - 1];
            const largestLayerName = largest.layerIdx === 0 ? 'W1 (Layer 0)' : 'W2 (Layer 1)';
            const largestLayer = models.layers[largest.layerIdx];
            const largestQuantized = largestLayer.data_flatten[largest.localIdx];
            const largestDequantized = (largestQuantized - largestLayer.center) * largestLayer.scale;

            console.log('Pruned weight range (by REAL magnitude):');
            console.log(`  Smallest: ${smallestLayerName}, local_idx=${smallest.localIdx}`);
            console.log(`    Before pruning - Quantized: ${smallestQuantized.toFixed(6)}, Real: ${smallestDequantized.toFixed(6)}`);
            console.log(`    After pruning  - Quantized: ${smallestLayer.center.toFixed(6)}, Real: 0.000000`);
            console.log(`  Largest:  ${largestLayerName}, local_idx=${largest.localIdx}`);
            console.log(`    Before pruning - Quantized: ${largestQuantized.toFixed(6)}, Real: ${largestDequantized.toFixed(6)}`);
            console.log(`    After pruning  - Quantized: ${largestLayer.center.toFixed(6)}, Real: 0.000000`);

            // Show distribution across layers
            let w1Count = 0, w2Count = 0;
            for (let i = 0; i < numToPrune; i++) {
                if (allWeights[i].layerIdx === 0) w1Count++;
                else if (allWeights[i].layerIdx === 1) w2Count++;
            }
            console.log(`Distribution: W1=${w1Count} weights, W2=${w2Count} weights`);
            console.log(`\n✓ Pruning sorted by REAL magnitudes and sets quantized values to 'center' (real = 0)`);
        }

        // Step 4: Mark the smallest weights for pruning
        const prunedIndices = [];
        for (let i = 0; i < numToPrune; i++) {
            prunedIndices.push(allWeights[i].globalIdx);
        }
        this.pruningIndices = prunedIndices.sort((a, b) => a - b);

        // Step 5: Create per-layer pruning masks
        for (let layerIdx = 0; layerIdx < models.layers.length; layerIdx++) {
            const layer = models.layers[layerIdx];
            if (!layer.data_flatten) {
                this.pruningMask.push(null);
                continue;
            }

            const mask = new Uint8Array(layer.data_flatten.length);

            // Mark weights to prune (1 = prune, 0 = keep)
            for (let i = 0; i < numToPrune; i++) {
                const entry = allWeights[i];
                if (entry.layerIdx === layerIdx) {
                    mask[entry.localIdx] = 1;
                }
            }

            this.pruningMask.push(mask);
        }

        // Debug: Verify bias exclusion
        if (this.pruningMask[0]) {
            let biasesPruned = 0;
            const [inChannels, outChannels] = models.layers[0].shape;
            const biasStart = (inChannels - 1) * outChannels;
            const biasEnd = inChannels * outChannels;

            for (let i = biasStart; i < biasEnd; i++) {
                if (this.pruningMask[0][i] === 1) biasesPruned++;
            }
            if (biasesPruned > 0) {
                console.error(`ERROR: ${biasesPruned} biases were marked for pruning!`);
            } else {
                console.log('✓ Biases correctly excluded from pruning');
            }
        }
    }

    generateNeuronPruningMask(models, fraction) {
        // Reset neuron pruning data structures
        this.neuronPruningMask = [];
        this.prunedNeuronIndices = [];

        if (fraction <= 0) {
            // No neuron pruning - initialize empty masks
            for (let i = 0; i < models.layers.length; i++) {
                const layer = models.layers[i];
                if (layer.data_flatten) {
                    this.neuronPruningMask.push(new Uint8Array(layer.data_flatten.length));
                } else {
                    this.neuronPruningMask.push(null);
                }
            }
            return;
        }

        // First pass: compute W1 column norms (input weights for each neuron)
        let w1ColNorms = null;
        if (models.layers[0] && models.layers[0].data_flatten) {
            const layer0 = models.layers[0];
            const data0 = layer0.data_flatten;
            const [rows0, cols0] = layer0.shape;  // [49, 96] including bias row
            const scale0 = layer0.scale || 1.0;
            const center0 = layer0.center || 0.0;
            const numInputRows = rows0 - 1;  // Exclude bias row (48 input rows)
            const numNeurons = cols0;  // 96 neurons

            w1ColNorms = new Array(numNeurons);
            for (let neuronIdx = 0; neuronIdx < numNeurons; neuronIdx++) {
                let sumSquares = 0;
                // Sum over input rows (excluding bias row)
                for (let inRow = 0; inRow < numInputRows; inRow++) {
                    const flatIdx = inRow * cols0 + neuronIdx;  // Row-major: column index
                    const quantized = data0[flatIdx];
                    const real = (quantized - center0) * scale0;  // DEQUANTIZE
                    sumSquares += real * real;
                }
                w1ColNorms[neuronIdx] = Math.sqrt(sumSquares);
            }
        }

        // Second pass: compute neuron masks based on combined norms
        for (let layerIdx = 0; layerIdx < models.layers.length; layerIdx++) {
            const layer = models.layers[layerIdx];

            if (!layer.data_flatten) {
                this.neuronPruningMask.push(null);
                continue;
            }

            const mask = new Uint8Array(layer.data_flatten.length);

            // Only prune neurons in Layer 1 (W2)
            if (layerIdx === 1) {
                const data = layer.data_flatten;
                const [numNeurons, outDim] = layer.shape;  // [96, 12]
                const scale = layer.scale || 1.0;
                const center = layer.center || 0.0;

                // Compute L2 norms for each neuron (row of W2 - output weights)
                const neuronNorms = [];
                for (let neuronIdx = 0; neuronIdx < numNeurons; neuronIdx++) {
                    let sumSquares = 0;
                    for (let outIdx = 0; outIdx < outDim; outIdx++) {
                        const flatIdx = neuronIdx * outDim + outIdx;  // Row-major indexing
                        const quantized = data[flatIdx];
                        const real = (quantized - center) * scale;  // DEQUANTIZE
                        sumSquares += real * real;
                    }
                    const w2Norm = Math.sqrt(sumSquares);

                    // Combine W1 column norm (input) and W2 row norm (output)
                    const combinedNorm = w1ColNorms ? (w1ColNorms[neuronIdx] + w2Norm) : w2Norm;

                    neuronNorms.push({
                        norm: combinedNorm,
                        w1Norm: w1ColNorms ? w1ColNorms[neuronIdx] : 0,
                        w2Norm: w2Norm,
                        neuronIdx: neuronIdx
                    });
                }

                // Sort by L2 norm (ascending - smallest first)
                neuronNorms.sort((a, b) => a.norm - b.norm);

                // Calculate number of neurons to prune
                const numToPrune = Math.round(fraction * numNeurons);

                console.log(`Neuron Pruning (Combined): ${numToPrune} / ${numNeurons} neurons (${(fraction * 100).toFixed(1)}%)`);

                if (numToPrune > 0) {
                    const smallest = neuronNorms[0];
                    const largestPruned = neuronNorms[numToPrune - 1];
                    console.log(`  Smallest combined norm: ${smallest.norm.toFixed(6)} (W1: ${smallest.w1Norm.toFixed(6)}, W2: ${smallest.w2Norm.toFixed(6)})`);
                    console.log(`  Largest pruned norm: ${largestPruned.norm.toFixed(6)} (W1: ${largestPruned.w1Norm.toFixed(6)}, W2: ${largestPruned.w2Norm.toFixed(6)})`);

                    // Create set of pruned neuron indices
                    const prunedSet = new Set();
                    for (let i = 0; i < numToPrune; i++) {
                        prunedSet.add(neuronNorms[i].neuronIdx);
                    }

                    // Mark all weights in pruned neuron rows
                    for (let neuronIdx = 0; neuronIdx < numNeurons; neuronIdx++) {
                        if (prunedSet.has(neuronIdx)) {
                            for (let outIdx = 0; outIdx < outDim; outIdx++) {
                                const flatIdx = neuronIdx * outDim + outIdx;
                                mask[flatIdx] = 1;
                            }
                        }
                    }

                    // Store pruned neuron indices for debugging
                    this.prunedNeuronIndices = Array.from(prunedSet).sort((a, b) => a - b);

                    // Also need to mark corresponding W1 columns (Layer 0)
                    // Go back and update Layer 0 mask if it exists
                    if (this.neuronPruningMask.length > 0 && this.neuronPruningMask[0]) {
                        const layer0 = models.layers[0];
                        const mask0 = this.neuronPruningMask[0];
                        const [rows0, cols0] = layer0.shape;  // [49, 96]

                        // Mark all elements in the pruned columns (for all rows)
                        for (let neuronIdx of prunedSet) {
                            for (let row = 0; row < rows0; row++) {
                                const flatIdx = row * cols0 + neuronIdx;
                                mask0[flatIdx] = 1;
                            }
                        }
                    }
                }
            }

            this.neuronPruningMask.push(mask);
        }
    }

    applyPruning(models) {
        // Clone the models
        const pruned = this.deepCloneModels(models);

        // Apply pruning mask to each layer
        for (let i = 0; i < pruned.layers.length; i++) {
            const layer = pruned.layers[i];
            const mask = this.pruningMask[i];

            if (layer.data_flatten && mask) {
                // To set real weight to zero, set quantized value to center
                // Because: real_value = (quantized_value - center) * scale
                // So: 0 = (center - center) * scale
                const center = layer.center || 0.0;

                for (let j = 0; j < layer.data_flatten.length; j++) {
                    if (mask[j] === 1) {
                        layer.data_flatten[j] = center;  // Set to center, NOT 0.0!
                    }
                }
            }
        }

        return pruned;
    }

    applyNeuronPruning(models) {
        // Clone the models
        const pruned = this.deepCloneModels(models);

        // Apply neuron pruning mask to each layer
        for (let i = 0; i < pruned.layers.length; i++) {
            const layer = pruned.layers[i];
            const mask = this.neuronPruningMask[i];

            if (layer.data_flatten && mask) {
                // CRITICAL: Set to center for true zero in dequantized space
                // Because: real_value = (quantized_value - center) * scale
                // So: 0 = (center - center) * scale
                const center = layer.center || 0.0;

                for (let j = 0; j < layer.data_flatten.length; j++) {
                    if (mask[j] === 1) {
                        layer.data_flatten[j] = center;  // Set to center, NOT 0.0!
                    }
                }
            }
        }

        return pruned;
    }

    updatePruningFraction(fraction) {
        this.weightPruningFraction = fraction;

        if (this.originalModels) {
            const gl = this.gl;

            // Regenerate pruning mask with new fraction
            this.generatePruningMask(this.originalModels, fraction);

            // Apply transformations in order: neuron pruning, weight pruning, then perturbations
            let modelsToUse = this.originalModels;

            if (this.neuronPruningFraction > 0) {
                modelsToUse = this.applyNeuronPruning(modelsToUse);
            }

            if (fraction > 0) {
                modelsToUse = this.applyPruning(modelsToUse);
            }

            if (this.weightPerturbation > 0) {
                modelsToUse = this.applyPerturbations(modelsToUse, this.weightPerturbation);
            }

            // Recreate GPU textures
            this.layers.forEach(layer => { if (layer.tex) gl.deleteTexture(layer.tex); });
            this.layers = modelsToUse.layers.map(layer => createDenseInfo(gl, layer));
        }
    }

    resetPruning() {
        if (this.originalModels) {
            console.log('Resetting pruning mask...');
            this.generatePruningMask(this.originalModels, this.weightPruningFraction);
            this.updatePruningFraction(this.weightPruningFraction);
        }
    }

    updateNeuronPruningFraction(fraction) {
        this.neuronPruningFraction = fraction;

        if (this.originalModels) {
            const gl = this.gl;

            // Regenerate neuron pruning mask with new fraction
            this.generateNeuronPruningMask(this.originalModels, fraction);

            // Apply transformations in order: neuron pruning, weight pruning, then perturbations
            let modelsToUse = this.originalModels;

            if (fraction > 0) {
                modelsToUse = this.applyNeuronPruning(modelsToUse);
            }

            if (this.weightPruningFraction > 0) {
                modelsToUse = this.applyPruning(modelsToUse);
            }

            if (this.weightPerturbation > 0) {
                modelsToUse = this.applyPerturbations(modelsToUse, this.weightPerturbation);
            }

            // Recreate GPU textures
            this.layers.forEach(layer => { if (layer.tex) gl.deleteTexture(layer.tex); });
            this.layers = modelsToUse.layers.map(layer => createDenseInfo(gl, layer));
        }
    }

    resetNeuronPruning() {
        if (this.originalModels) {
            console.log('Resetting neuron pruning mask...');
            this.generateNeuronPruningMask(this.originalModels, this.neuronPruningFraction);
            this.updateNeuronPruningFraction(this.neuronPruningFraction);
        }
    }

    setAltWeights(models) {
        const gl = this.gl;

        // Check if alt model type matches primary model type
        const altModelType = models.model_type || detectModelType(models, this.channelCount);
        const altIsRD = (altModelType === 'rd');

        if (altIsRD !== this.isRD) {
            console.warn(`Warning: Alt model type (${altModelType}) differs from primary model type (${this.isRD ? 'rd' : 'nca'}). Blending may not work correctly.`);
        }

        this.altLayers.forEach(layer => { if (layer.tex) gl.deleteTexture(layer.tex); });
        this.altLayers = models.layers.map(layer => createDenseInfo(gl, layer));
    }

    runLayer(program, output, inputs) {
        const gl = this.gl;
        inputs = inputs || {};
        const uniforms = {};
        for (const name in inputs) {
            const val = inputs[name];
            if (val._type == 'tensor') {
                setTensorUniforms(uniforms, name, val);
            } else {
                uniforms[name] = val;
            }
        }
        uniforms['u_shuffleTex'] = this.shuffleTex;
        uniforms['u_shuffleOfs'] = this.shuffleOfs;
        uniforms['HW'] = this.gridSize;
        setTensorUniforms(uniforms, 'u_output', output);

        twgl.bindFramebufferInfo(gl, output.fbi);
        gl.useProgram(program.program);
        twgl.setBuffersAndAttributes(gl, program, this.quad);
        twgl.setUniforms(program, uniforms);
        twgl.drawBufferInfo(gl, this.quad);
        return {programName: program.name, output}
    }

    runDense(output, input, layer, relu = false, seed = 0) {
        return this.runLayer(this.progs.dense, output, {
            u_input: input, u_control: this.buf.control,
            u_weightTex: layer.tex, u_weightCoefs: layer.coefs, u_layout: layer.layout,
            u_seed: seed, u_fuzz: this.fuzz, u_updateProbability: this.updateProbability,
            bias: layer.bias, pos_emb: layer.pos_emb, relu: relu,
            grid_size: this.gridSize, u_angle: this.rotationAngle / 180.0 * Math.PI,
            u_isRD: this.isRD ? 1.0 : 0.0,
        });
    }

    draw(zoom, viewChannel) {
        const gl = this.gl;
        zoom = zoom || 1.0;
        viewChannel = viewChannel !== undefined ? viewChannel : -1.0;  // default RGB

        // Update visualization scale/offset if auto-scale is enabled
        this.updateVisScaleOffset();

        gl.useProgram(this.progs.vis.program);
        twgl.setBuffersAndAttributes(gl, this.progs.vis, this.quad);
        const uniforms = {
            u_raw: 0.0, u_zoom: zoom,
            u_angle: this.rotationAngle / 180.0 * Math.PI,
            u_alignment: this.alignment,
            u_perceptionCircle: this.perceptionCircle,
            u_arrows: this.arrowsCoef,
            u_hexGrid: this.hexGrid,
            u_viewChannel: viewChannel,
            u_invertColors: this.invertColors ? 1.0 : 0.0,
            u_isRotInv: (this.isRotInv || this.isSqzast) ? 1.0 : 0.0,
            u_thetaChannel: this.thetaChannel,
            u_growingMode: this.growingMode ? 1.0 : 0.0,
            u_visScale: this.visScaleOffset.scale,
            u_visOffset: this.visScaleOffset.offset,
        };
        let inputBuf = this.buf.state;
        if (this.visMode != 'color') {
            inputBuf = this.buf[this.visMode];
            uniforms.u_raw = 0.0;
        }
        inputBuf = this.buf.state;
        setTensorUniforms(uniforms, 'u_input', inputBuf);
        twgl.setUniforms(this.progs.vis, uniforms);
        twgl.drawBufferInfo(gl, this.quad);
    }
}