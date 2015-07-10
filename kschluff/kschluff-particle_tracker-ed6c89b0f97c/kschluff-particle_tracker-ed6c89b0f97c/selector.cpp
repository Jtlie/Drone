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
#include "selector.h"
#include <cmath>
using namespace cv;

void Selector::mouse_callback(int event, int x, int y, int flags, void* data)
{
   Selector& self = *((Selector*)data); 
   
   switch( event )
   {
      case CV_EVENT_LBUTTONDOWN:
	 self.m_selection_valid = false;
	 self.m_selecting = true;
	 self.m_selection = Rect(0,0,0,0);
	 self.m_origin.x = x;
	 self.m_origin.y = y;
	 break;

      case CV_EVENT_LBUTTONUP:
	 self.m_selection_valid = true;
	 self.m_selecting = false;

      default:
	 if( self.m_selecting )
	 {
	    self.m_selection.x = MIN(x, self.m_origin.x);
	    self.m_selection.y = MIN(y, self.m_origin.y);
	    self.m_selection.width = std::abs(x - self.m_origin.x);
	    self.m_selection.height = std::abs(y - self.m_origin.y);
	 }
   }
}
