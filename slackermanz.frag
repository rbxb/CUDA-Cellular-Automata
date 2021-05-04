vec3 nhd( ivec2 nbhd, ivec2 ofst, float psn, float thr, int col ) {
    // Neighbourhood: Return information about the specified group of pixels
    float c_total = 0.0;
    float c_valid = 0.0;
    float c_value = 0.0;

    for(float i = -nbhd[0]; i <= nbhd[0]; i+=1.0) {
        for(float j = -nbhd[0]; j <= nbhd[0]; j+=1.0) {

            float dist = round(sqrt(i*i+j*j));

            if( dist <= nbhd[0] && dist > nbhd[1] && dist != 0.0 ) {
                float cval = gdv(ivec2(i+ofst[0],j+ofst[1]),col);
                float c_total += psn;

                if( cval > thr ) {
                    c_valid += psn;
                    cval = psn * cval;
                    c_value += cval-fract(cval);
                }
            }
        }
    }
    return vec3( c_value, c_valid, c_total );
}

// nbhd (outer radius, inner radius)
// ofst offset from origin
// psn precision limit
// thr threshold; only consider cells with a value of this or higher
// col color channel to look at