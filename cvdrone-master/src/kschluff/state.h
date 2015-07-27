#ifndef STATE_H
#define STATE_H
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

struct StateData;
struct State_;
typedef State_ (*State)(StateData&);
struct State_
{
   State_( State pp ) : p( pp ) { }
   operator State() { return p; }
   State p;
};

State_ state_start(StateData& d);
State_ state_selecting(StateData& d);
State_ state_initializing(StateData& d);
State_ state_tracking(StateData& d);

#endif
