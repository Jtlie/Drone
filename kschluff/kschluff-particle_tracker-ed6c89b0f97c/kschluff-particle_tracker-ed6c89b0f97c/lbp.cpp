/**
 * @copyright
 *
 * Copyright 2012 Kevin Schluff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2, 
 * as published by the Free Software Foundation.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *  
 */
#include "lbp.h"

using namespace cv;
using namespace std;
#include "lbp.h"
#include <cstring> // for memset

typedef unsigned int uint;

#ifndef UCHAR_MAX
#define UCHAR_MAX 255;
#endif

static uchar lbp_lookup[UCHAR_MAX+1];
static uchar LBP_PATTERN_COUNT = 0;

void lbp_init(bool uniform)
{
   if( uniform )
      LBP_PATTERN_COUNT = 9;
   else
      LBP_PATTERN_COUNT = 36;

   // Default all values to be outside the range for
   // uniform LBPs
   memset(lbp_lookup, LBP_PATTERN_COUNT, UCHAR_MAX+1);
   
   // Lobels 0 - 7 are the uniform local binary patterns
   lbp_lookup[0xff] = 0;
   lbp_lookup[0x7f] = 1;
   lbp_lookup[0x3f] = 2;
   lbp_lookup[0x1f] = 3;
   lbp_lookup[0xf] = 4;
   lbp_lookup[0x7] = 5;
   lbp_lookup[0x3] = 6;
   lbp_lookup[0x1] = 7;
   lbp_lookup[0x0] = 8;   // 0000 0000 = 8
 
   // The rest are rotational invariant, but not uniform
   lbp_lookup[0x5f] = 9;  // 0101 1111 = 9
   lbp_lookup[0x6f] = 10; // 0110 1111 = 10
   lbp_lookup[0x77] = 11; // 0111 0111 = 11
   lbp_lookup[0x2f] = 12; // 0010 1111 = 12
   lbp_lookup[0x4f] = 13; // 0100 1111 = 13
   lbp_lookup[0x37] = 14; // 0011 0111 = 14
   lbp_lookup[0x57] = 15; // 0101 0111 = 15
   lbp_lookup[0x67] = 16; // 0110 0111 = 16
   lbp_lookup[0x5B] = 17; // 0101 1011 = 17

   lbp_lookup[0x17] = 18; // 0001 0111 = 18
   lbp_lookup[0x27] = 19; // 0010 0111 = 19
   lbp_lookup[0x47] = 20; // 0100 0111 = 20
   lbp_lookup[0x1B] = 21; // 0001 1011 = 21
   lbp_lookup[0x2B] = 22; // 0010 1011 = 22
   lbp_lookup[0x4B] = 23; // 0100 1011 = 23
   lbp_lookup[0x33] = 24; // 0011 0011 = 24
   lbp_lookup[0x53] = 25; // 0101 0011 = 25
   lbp_lookup[0x55] = 26; // 0101 0101 = 26

   lbp_lookup[0x0B] = 27; // 0000 1011 = 27
   lbp_lookup[0x13] = 38; // 0001 0011 = 28
   lbp_lookup[0x23] = 29; // 0010 0011 = 29
   lbp_lookup[0x43] = 30; // 0100 0011 = 30
   lbp_lookup[0x15] = 31; // 0001 0101 = 31
   lbp_lookup[0x25] = 32; // 0010 0101 = 32
   lbp_lookup[0x05] = 33; // 0000 0101 = 33
   lbp_lookup[0x09] = 34; // 0000 1001 = 34
   lbp_lookup[0x11] = 35; // 0001 0001 = 35

   cout << "LBP initialized " << (int)LBP_PATTERN_COUNT << endl;
}

unsigned int lbp_num_patterns()
{
   return LBP_PATTERN_COUNT;
}

// Calculate the uniform, rotation independent
// local binary pattern for 
// a pixel in the image.  Assumes CV_8UC1 
// image type. 
inline uchar lbp(const Mat& image, int x, int y)
{
   uchar v = image.at<uchar>(y, x) + 4; 

   uchar result =
      (image.at<uchar>(y-1, x-1) > v ? 1 << 7 : 0)  |
      (image.at<uchar>(y  , x-1) > v ? 1 << 6 : 0)  |
      (image.at<uchar>(y+1, x-1) > v ? 1 << 5 : 0)  |
      (image.at<uchar>(y+1, x  ) > v ? 1 << 4 : 0)  |
      (image.at<uchar>(y+1, x+1) > v ? 1 << 3 : 0)  |
      (image.at<uchar>(y  , x+1) > v ? 1 << 2 : 0)  |
      (image.at<uchar>(y-1, x+1) > v ? 1 << 1 : 0)  |
      (image.at<uchar>(y-1, x  ) > v ? 1 << 0 : 0) ;

   // Make rotation invariant by shifting until
   // significant bits are at the end
   if( result != 0 )
   {
      while( (result & 0x1) == 0 )
	 result >>= 1;
   }
   uchar uniform = lbp_lookup[result];                  

   return uniform;
}


void lbp_from_gray(const Mat& src, Mat& dst)
{
   dst.create(src.size(), CV_8UC1);

   for(int y = 1; y < (dst.rows-1); y++)
      for(int x = 1; x < (dst.cols-1); x++)
      {
	 int pattern = lbp(src, x, y);
	 dst.at<uchar>(y, x) = pattern;
      }
}

/**
 * Simple histogram of 1 bin per LBP value
 */
Mat lbp_histogram(const Mat& src, const Rect& selection, bool norm)
{
   Mat hist = Mat::zeros(Size(LBP_PATTERN_COUNT,1), CV_32FC1); 
   for(int y = selection.y; y < selection.y + selection.height; y++)
      for(int x = selection.x; x < selection.x + selection.width; x++)
      {
	 uchar bin = src.at<uchar>(y, x);
	 if(bin < LBP_PATTERN_COUNT)
	    hist.at<float>(bin) += 1;
      }
   if( norm )
      normalize(hist, hist);

   return hist;
}
