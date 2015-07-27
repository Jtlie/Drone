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
#include "hist.h"
#include "lbp.h"

using namespace cv;
using namespace std;

inline void calc_hist_bgr(Mat& bgr, Mat& lbp, Mat& hist)
{
   static const int channels[] = {0, 1, 2}; 
   static const int b_bins = 8; 
   static const int g_bins = 8; 
   static const int r_bins = 8;
   static const int hist_size[] = {b_bins, g_bins, r_bins};
   static const float branges[] = {0, 255};
   static const float granges[] = {0, 255};
   static const float rranges[] = {0, 255};
   static const float* ranges[] = {branges, granges, rranges};
   static const Mat mask;
   static const int dims = 3;
   Mat srcs[] = {bgr};

   calcHist(srcs, sizeof(srcs), channels, mask, hist, dims, hist_size, ranges, true, false);
}

inline void calc_hist_bgrl(Mat& bgr, Mat& lbp, Mat& hist)
{
   static const int channels[] = {0, 1, 2, 3}; 
   static const int b_bins = 8; 
   static const int g_bins = 8; 
   static const int r_bins = 8;
   static const int l_bins = lbp_num_patterns();
   static const int hist_size[] = {b_bins, g_bins, r_bins, l_bins};
   static const float branges[] = {0, 255};
   static const float granges[] = {0, 255};
   static const float rranges[] = {0, 255};
   static const float lranges[] = {0, lbp_num_patterns()};
   static const float* ranges[] = {branges, granges, rranges, lranges};
   static const Mat mask;
   static const int dims = 4;
   Mat srcs[] = {bgr, lbp};

   calcHist(srcs, sizeof(srcs), channels, mask, hist, dims, hist_size, ranges, true, false);
}

void calc_hist(cv::Mat& bgr, cv::Mat& lbp, cv::Mat& hist, bool use_lbp)
{
   if( use_lbp )
   {
      return calc_hist_bgrl(bgr, lbp, hist);
   }
   else
   {
      return calc_hist_bgr(bgr, lbp, hist);
   }
}
