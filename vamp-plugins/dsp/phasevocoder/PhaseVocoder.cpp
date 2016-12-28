/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    QM DSP Library

    Centre for Digital Music, Queen Mary, University of London.
    This file 2005-2006 Christian Landone, copyright 2013 QMUL.

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version.  See the file
    COPYING included with this distribution for more information.
*/

#include "PhaseVocoder.h"
#include "dsp/transforms/FFT.h"
#include "maths/MathUtilities.h"
#include <math.h>

#include <cassert>

#include <iostream>

#ifdef __ARM_NEON__
#include "arm_neon.h"
#endif

using std::cerr;
using std::endl;

PhaseVocoder::PhaseVocoder(int n, int hop) :
    m_n(n),
    m_hop(hop)
{
    m_fft = new FFTReal(m_n);
    m_time = new fl_t[m_n];
    m_real = new fl_t[m_n];
    m_imag = new fl_t[m_n];
    m_phase = new fl_t[m_n/2 + 1];
    m_unwrapped = new fl_t[m_n/2 + 1];

    for (int i = 0; i < m_n/2 + 1; ++i) {
        m_phase[i] = 0.0;
        m_unwrapped[i] = 0.0;
    }

    reset();
}

PhaseVocoder::~PhaseVocoder()
{
    delete[] m_unwrapped;
    delete[] m_phase;
    delete[] m_real;
    delete[] m_imag;
    delete[] m_time;
    delete m_fft;
}

void PhaseVocoder::FFTShift(const fl_t *src)
{
    const int hs = m_n/2;
    for (int i = 0; i < hs; ++i) {
        m_time[i] = src[i + hs];
        m_time[i + hs] = src[i];
    }
}

void PhaseVocoder::processTimeDomain(const fl_t *src,
                                     fl_t *mag, fl_t *theta,
                                     fl_t *unwrapped)
{
    FFTShift(src);
    m_fft->forward(m_time, m_real, m_imag);
    getMagnitudes(mag);
    getPhases(theta);
    unwrapPhases(theta, unwrapped);
}

void PhaseVocoder::processFrequencyDomain(const fl_t *reals, 
                                          const fl_t *imags,
                                          fl_t *mag, fl_t *theta,
                                          fl_t *unwrapped)
{
    for (int i = 0; i < m_n/2 + 1; ++i) {
        m_real[i] = reals[i];
        m_imag[i] = imags[i];
    }
    getMagnitudes(mag);
    getPhases(theta);
    unwrapPhases(theta, unwrapped);
}

void PhaseVocoder::reset()
{
    for (int i = 0; i < m_n/2 + 1; ++i) {
        // m_phase stores the "previous" phase, so set to one step
        // behind so that a signal with initial phase at zero matches
        // the expected values. This is completely unnecessary for any
        // analytical purpose, it's just tidier.
        fl_t omega = (2 * M_PI * m_hop * i) / m_n;
        m_phase[i] = -omega;
        m_unwrapped[i] = -omega;
    }
}

void PhaseVocoder::getMagnitudes(fl_t *mag)
{
    int i = 0;
    int count = m_n/2 + 1;
#if defined(VAMP_FLOAT_MATH) && defined(__ARM_NEON__)
    for (; i < count-3; i+=4) {
        // get real/imag
        float32x4_t vreal = vld1q_f32(m_real+i);
        float32x4_t vimag = vld1q_f32(m_imag+i);
        // sum^2
        float32x4_t vsum = (vreal * vreal + vimag * vimag);
        // save
        vst1q_f32(mag+i, vsum);
    }
    for (i = 0; i < count-3; i+=4) {
        mag[i+0] = sqrt(mag[i+0]);
        mag[i+1] = sqrt(mag[i+1]);
        mag[i+2] = sqrt(mag[i+2]);
        mag[i+3] = sqrt(mag[i+3]);
    }
#endif
    for (; i < count; i++) {
        mag[i] = sqrt(m_real[i] * m_real[i] + m_imag[i] * m_imag[i]);
    }
}

void PhaseVocoder::getPhases(fl_t *theta)
{
    for (int i = 0; i < m_n/2 + 1; i++) {
	theta[i] = atan2(m_imag[i], m_real[i]);
    }	
}

void PhaseVocoder::unwrapPhases(fl_t *theta, fl_t *unwrapped)
{
    for (int i = 0; i < m_n/2 + 1; ++i) {

        fl_t omega = (2 * M_PI * m_hop * i) / m_n;
        fl_t expected = m_phase[i] + omega;
        fl_t error = MathUtilities::princarg(theta[i] - expected);

        unwrapped[i] = m_unwrapped[i] + omega + error;

        m_phase[i] = theta[i];
        m_unwrapped[i] = unwrapped[i];
    }
}

