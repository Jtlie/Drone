#ifndef SELECTOR_H
#define SELECTOR_H
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

#include <opencv2/opencv.hpp>

class Selector
{
public:

   Selector(const char* window)
      :m_selection_valid(false),
       m_selecting(false),
       m_selection(),
       m_origin()
   {
      cv::setMouseCallback(window, mouse_callback, this);
   }

   bool valid() const
   { return m_selection_valid; }
   
   bool selecting() const
   { return m_selecting; }

   const cv::Rect& selection() const
   { return m_selection; }

private:
   static void mouse_callback(int event, int x, int y, int flags, void* data);

private:
   bool m_selection_valid;
   bool m_selecting;
   cv::Rect m_selection;
   cv::Point m_origin;
};

#endif
