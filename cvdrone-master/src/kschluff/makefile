UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
# A quirk of my Mac setup, you probably won't need this
PKG_CONFIG = /usr/local/bin/pkg-config
else
PKG_CONFIG = pkg-config
endif

CFLAGS = -O2 -Wall `$(PKG_CONFIG) opencv --cflags`
LIBS =    `$(PKG_CONFIG) opencv --libs`
SRCS = main.cpp condens.cpp lbp.cpp selector.cpp filter.cpp hist.cpp
HEADERS =  condens.h lbp.h selector.h filter.h state.h hist.h

particle_tracker: $(SRCS) $(HEADERS)
	g++ $(CFLAGS) -o particle_tracker $(LIBS) $(SRCS)

.PHONY clean:
	rm -f particle_tracker particle_tracker.exe
