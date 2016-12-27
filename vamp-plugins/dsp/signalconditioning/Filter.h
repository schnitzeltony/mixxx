/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    QM DSP Library

    Centre for Digital Music, Queen Mary, University of London.
    This file 2005-2006 Christian Landone.

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version.  See the file
    COPYING included with this distribution for more information.
*/

#ifndef FILTER_H
#define FILTER_H

#include "maths/MathAliases.h"

#ifndef NULL
#define NULL 0
#endif

/**
 * Filter specification. For a filter of order ord, the ACoeffs and
 * BCoeffs arrays must point to ord+1 values each. ACoeffs provides
 * the denominator and BCoeffs the numerator coefficients of the
 * filter.
 */
struct FilterConfig{
    unsigned int ord;
    fl_t* ACoeffs;
    fl_t* BCoeffs;
};

/**
 * Digital filter specified through FilterConfig structure.
 */
class Filter  
{
public:
    Filter( FilterConfig Config );
    virtual ~Filter();

    void reset();

    void process( fl_t *src, fl_t *dst, unsigned int length );

private:
    void initialise( FilterConfig Config );
    void deInitialise();

    unsigned int m_ord;

    fl_t* m_inBuffer;
    fl_t* m_outBuffer;

    fl_t* m_ACoeffs;
    fl_t* m_BCoeffs;
};

#endif
