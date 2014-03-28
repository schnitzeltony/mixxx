#ifndef CONTROLEFFECTKNOB_H
#define CONTROLEFFECTKNOB_H

#include "controlpotmeter.h"

class ControlEffectKnob : public ControlPotmeter {
    Q_OBJECT
  public:
    ControlEffectKnob(ConfigKey key, double dMinValue = 0.0, double dMaxValue = 1.0);
    void setType(double type);
  private:
    double m_type;
};

#endif // CONTROLLEFFECTKNOB_H