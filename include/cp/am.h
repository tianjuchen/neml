#pragma once

#include "slipharden.h"

#include "../objects.h"
#include "../history.h"
#include "../interpolate.h"

#include "../windows.h"

namespace neml {

class NEML_EXPORT AMModel: public SlipHardening
{
 public:
  LANLTiModel(ParameterSet & params);

  /// String type for the object system
  static std::string type();
  /// Initialize from a parameter set
  static std::unique_ptr<NEMLObject> initialize(ParameterSet & params);
  /// Default parameters
  static ParameterSet parameters();

  /// Report your variable names
  virtual std::vector<std::string> varnames() const;
  /// Set new varnames
  virtual void set_varnames(std::vector<std::string> vars);

  /// Request whatever history you will need
  virtual void populate_hist(History & history) const;
  /// Setup history
  virtual void init_hist(History & history) const;

  /// Map the set of history variables to the slip system hardening
  virtual double hist_to_tau(size_t g, size_t i, const History & history,
                             Lattice & L,
                             double T, const History & fixed) const;
  /// Derivative of the map wrt to history
  virtual History
      d_hist_to_tau(size_t g, size_t i, const History & history, Lattice & L,
                    double T, const History & fixed) const;

  /// The rate of the history
  virtual History hist(const Symmetric & stress,
                     const Orientation & Q, const History & history,
                     Lattice & L, double T, const SlipRule & R,
                     const History & fixed) const;
  /// Derivative of the history wrt stress
  virtual History d_hist_d_s(const Symmetric & stress,
                             const Orientation & Q, const History & history,
                             Lattice & L, double T,
                             const SlipRule & R,
                             const History & fixed) const;
  /// Derivative of the history wrt the history
  virtual History
      d_hist_d_h(const Symmetric & stress,
                 const Orientation & Q,
                 const History & history,
                 Lattice & L,
                 double T, const SlipRule & R,
                 const History & fixed) const;
  /// Derivative of this history wrt the history, external variables
  virtual History
      d_hist_d_h_ext(const Symmetric & stress,
                     const Orientation & Q,
                     const History & history,
                     Lattice & L,
                     double T, const SlipRule & R,
                     const History & fixed,
                     std::vector<std::string> ext) const;

 protected:
  size_t nadi_() const {return dc_.size();};
  size_t nslip_() const {return kw2_.size();};
  size_t wslip_() const {return kw1_.size();};
  size_t islip_() const {return ki1_.size();};
  size_t size() const {return kw1_.size() + ki2_.size() + dc_.size();};
  void consistency(Lattice & L) const;

 private:
  std::vector<std::shared_ptr<Interpolate>> mu_, kw1_, kw2_, ki1_, ki2_;
  // std::shared_ptr<SquareMatrix> C_st_;
  std::vector<std::shared_ptr<Interpolate>> mu_, k1_, k2_;
  double alpha_w_, alpha_i_, iniwvalue_, iniivalue_, inibvalue_;
  double kb_, R_, k0_, dc_, c_, lambda_, omega_, Q_;
  double Tr_, ftr_;
  std::string varprefix_, wslipprefix, islipprefix;
  std::vector<std::string> varnames_;
};

static Register<AMModel> regAMModel;


} // namespace neml