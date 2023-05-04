#include "cp/addmaf.h"

namespace neml
{

AMModel::AMModel(ParameterSet & params):
    SlipHardening(params),
	mu_(params.get_object_parameter_vector<Interpolate>("mu")),
    kw1_(params.get_object_parameter_vector<Interpolate>("kw1")), 
    kw2_(params.get_object_parameter_vector<Interpolate>("kw2")),
    ki1_(params.get_object_parameter_vector<Interpolate>("ki1")), 
    ki2_(params.get_object_parameter_vector<Interpolate>("ki2")),
	alpha_w_(params.get_parameter<double>("alpha_w")),
	alpha_i_(params.get_parameter<double>("alpha_i")),
	iniwvalue_(params.get_parameter<double>("iniwvalue")),
	iniivalue_(params.get_parameter<double>("iniivalue")),
	inibvalue_(params.get_parameter<double>("inibvalue")),
	b_(params.get_parameter<double>("b")),
	kb_(params.get_parameter<double>("kb")),
	R_(params.get_parameter<double>("R")),
	k0_(params.get_parameter<double>("k0")),
	dc_(params.get_parameter<double>("dc")),
	c_(params.get_parameter<double>("c")),
	lambda_(params.get_parameter<double>("lamda")),
	omega_(params.get_parameter<double>("omega")),
	Q_(params.get_parameter<double>("Q")),
	Tr_(params.get_parameter<double>("Tr")),
	ftr_(params.get_parameter<double>("ftr")),
	initsigma_(params.get_parameter<double>("initsigma")),
	varprefix_(params.get_parameter<std::string>("varprefix")), 
    wslipprefix_(params.get_parameter<std::string>("wslipprefix")),
	islipprefix_(params.get_parameter<std::string>("islipprefix"))
{ 

  if (kw1_.size() != nslip_() and kw2_.size() != nslip_()) {
    throw std::invalid_argument("Dislocation systems and slip systems do not agree!");
  }

  varnames_.resize(size());
  for (size_t i = 0; i < size(); i++) {
    if (i < nadi_()) {
      varnames_[i] = varprefix_+std::to_string(i);
    } 
	else if (i >= nadi_() and i < wslip_() + nadi_()){
	  varnames_[i] = wslipprefix_+std::to_string(i);
	}
    else {
      varnames_[i] = islipprefix_+std::to_string(i);	
    }
  }
  init_cache_();
}

std::string AMModel::type()
{
  return "AMModel";
}

std::unique_ptr<NEMLObject> AMModel::initialize(ParameterSet & params)
{
  return neml::make_unique<AMModel>(params);
}

ParameterSet AMModel::parameters()
{
  ParameterSet pset(AMModel::type());

  pset.add_parameter<std::vector<NEMLObject>>("mu");
  pset.add_parameter<std::vector<NEMLObject>>("kw1");
  pset.add_parameter<std::vector<NEMLObject>>("kw2");
  pset.add_parameter<std::vector<NEMLObject>>("ki1");
  pset.add_parameter<std::vector<NEMLObject>>("ki2");
  pset.add_optional_parameter<double>("alpha_w", 0.95);
  pset.add_optional_parameter<double>("alpha_i", 0.25);
  pset.add_optional_parameter<double>("iniwvalue", 5.0e-6);
  pset.add_optional_parameter<double>("iniivalue", 1.0e-8);
  pset.add_optional_parameter<double>("inibvalue", 5.5e2);
  pset.add_optional_parameter<double>("b", 0.256);
  pset.add_optional_parameter<double>("kb", 13806.49);
  pset.add_optional_parameter<double>("R", 8.3145);
  pset.add_optional_parameter<double>("k0", 1.0e-6);
  pset.add_optional_parameter<double>("dc", 1.0e4);
  pset.add_optional_parameter<double>("c", 10.0);
  pset.add_optional_parameter<double>("lamda", 1.0);
  pset.add_optional_parameter<double>("omega", 421750.0);
  pset.add_optional_parameter<double>("Q", 1.0e4);
  pset.add_optional_parameter<double>("Tr", 298.0);
  pset.add_optional_parameter<double>("ftr", 0.1);
  pset.add_optional_parameter<double>("initsigma", 50.0);
  pset.add_optional_parameter<std::string>("varprefix", 
                                           std::string("wall"));
  pset.add_optional_parameter<std::string>("wslipprefix", 
                                           std::string("wslip"));
  pset.add_optional_parameter<std::string>("islipprefix", 
                                           std::string("islip"));
  return pset;
}

std::vector<std::string> AMModel::varnames() const
{
  return varnames_;
}

void AMModel::set_varnames(std::vector<std::string> vars)
{
  varnames_ = vars;
  init_cache_();
}

void AMModel::populate_hist(History & history) const
{
  for (auto vn : varnames_) {
    history.add<double>(vn);
  }
}

void AMModel::init_hist(History & history) const
{
  for (size_t i = 0; i < size(); i++) {
    if (i < nadi_()) {
      history.get<double>(varnames_[i]) = inibvalue_; 
    } 
	else if (i >= nadi_() and i < wslip_() + nadi_()){
	  history.get<double>(varnames_[i]) = iniwvalue_;
	}
    else {
      history.get<double>(varnames_[i]) = iniivalue_;
    }
  }
}

double AMModel::wall_frac(size_t g, size_t i, 
                                const History & history,
                                Lattice & L,
                                double T, const History & fixed) const
{
  consistency(L);
  
  return (omega_ * mu_[L.flat(g,i)]->value(T) * std::pow(b_, 3))
	/ (kb_ * T * history.get<double>(varnames_[0]));
}


double AMModel::fmod(const History & history) const
{
  return 1 / (1 + std::exp(-c_ * (history.get<double>(varnames_[0]) / dc_ - 1.0)));
}


double AMModel::macaulay(double x) const
{
  return (x + std::fabs(x)) / 2.0;
}

double AMModel::hist_to_tau(size_t g, size_t i, 
                                const History & history,
                                Lattice & L,
                                double T, const History & fixed) const
{
  consistency(L);

  return alpha_i_ * mu_[L.flat(g,i)]->value(T) * b_ 
	* std::sqrt(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_() + islip_()])))
	* (1 - wall_frac(g, i, history, L, T, fixed) * (1 - fmod(history)))
	+ alpha_w_ * mu_[L.flat(g,i)]->value(T) * b_
	* std::sqrt(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_()])))
	* wall_frac(g, i, history, L, T, fixed) * (1 - fmod(history))
	+ initsigma_;
}  


double AMModel::dfdd(size_t g, size_t i, 
						  const History & history,
						  Lattice & L,
						  double T) const
{
  consistency(L);

  return -mu_[L.flat(g,i)]->value(T) * omega_ * std::pow(b_, 3)
	/ (kb_ * T * std::pow(history.get<double>(varnames_[0]), 2));
}


double AMModel::dfsigdd(const History & history) const
{  
  return c_ / dc_ * std::exp(c_ - c_ / dc_ * history.get<double>(varnames_[0]))
	/ std::pow((std::exp(c_ - c_ / dc_ * history.get<double>(varnames_[0])) + 1), 2);
}


History AMModel::d_hist_to_tau(size_t g, size_t i, 
                                   const History & history,
                                   Lattice & L,
                                   double T, 
                                   const History & fixed) const
{
  consistency(L);
  History res = cache(CacheType::DOUBLE);

  res.get<double>(varnames_[0]) = alpha_i_ * mu_[L.flat(g,i)]->value(T) * b_
	* std::sqrt(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_() + islip_()])))
	* (-dfdd(g, i, history, L, T) + dfdd(g, i, history, L, T) * fmod(history)
	+ wall_frac(g, i, history, L, T, fixed) * dfsigdd(history))
	+ alpha_w_ * mu_[L.flat(g,i)]->value(T) * b_
	* std::sqrt(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_()])))
	* (dfdd(g, i, history, L, T) - (dfdd(g, i, history, L, T) * fmod(history)
	+ wall_frac(g, i, history, L, T, fixed) * dfsigdd(history))); 


  res.get<double>(varnames_[L.flat(g,i) + nadi_()]) = 0.5 * alpha_w_ * mu_[L.flat(g,i)]->value(T) * b_
	* 1 / std::sqrt(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_()])))
	* wall_frac(g, i, history, L, T, fixed) * (1 - fmod(history));

  res.get<double>(varnames_[L.flat(g,i) + islip_() + nadi_()]) = 0.5 * alpha_i_ * mu_[L.flat(g,i)]->value(T) * b_
	* 1 / std::sqrt(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_() + islip_()])))
	* (1 - wall_frac(g, i, history, L, T, fixed) * (1 - fmod(history)));

  return res;
}

History AMModel::hist(const Symmetric & stress, 
                          const Orientation & Q,
                          const History & history, 
                          Lattice & L, double T, const SlipRule & R, 
                          const History & fixed) const
{
  consistency(L); 

  History res = blank_hist();
  
  double ddot = k0_ * std::exp(-Q_ /( R_ * T))
	* history.get<double>(varnames_[0])
	* std::exp(-history.get<double>(varnames_[0])/dc_);

  for (size_t g = 0; g < L.ngroup(); g++) {
    for (size_t i = 0; i < L.nslip(g); i++) {
      size_t k = L.flat(g,i);
	    res.get<double>(varnames_[0]) = k0_ * std::exp(-Q_ /( R_ * T))
			* history.get<double>(varnames_[0])
			* std::exp(-history.get<double>(varnames_[0])/dc_);
        res.get<double>(varnames_[k + nadi_()]) = 
            (kw1_[k]->value(T) * std::sqrt(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_()])))
			- kw2_[k]->value(T) * macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_()])))
			* fabs(R.slip(g,i,stress,Q,history,L,T,fixed))
			- 2 / lambda_ * std::pow(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_()])), 1.5)
			* k0_ * std::exp(-Q_ /( R_ * T))
			* history.get<double>(varnames_[0])
			* std::exp(-history.get<double>(varnames_[0])/dc_)
			* T / (T + Tr_) * ftr_;
			
		res.get<double>(varnames_[k + nadi_() + islip_()]) = 
            (ki1_[k]->value(T) * std::sqrt(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_() + islip_()])))
			- ki2_[k]->value(T) * macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_() + islip_()])))
			* fabs(R.slip(g,i,stress,Q,history,L,T,fixed))
			+ 2 / lambda_ * std::pow(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_()])), 1.5)
			* k0_ * std::exp(-Q_ /( R_ * T))
			* history.get<double>(varnames_[0])
			* std::exp(-history.get<double>(varnames_[0])/dc_)
			* T / (T + Tr_) * ftr_;
      
    }
  }
  return res;
}

History AMModel::d_hist_d_s(const Symmetric & stress, 
                                const Orientation & Q, 
                                const History & history,
                                Lattice & L, double T, 
                                const SlipRule & R,
                                const History & fixed) const
{
  consistency(L);
  History res = blank_hist().derivative<Symmetric>();
  
  res.get<Symmetric>(varnames_[0]) = Symmetric::zero();
  
  for (size_t g = 0; g < L.ngroup(); g++) {
    for (size_t i = 0; i < L.nslip(g); i++) { 
      size_t k = L.flat(g,i);
	  double slip = R.slip(g, i, stress, Q, history, L, T, fixed); 
	  res.get<Symmetric>(varnames_[k + nadi_()]) = 
            (kw1_[k]->value(T) * std::sqrt(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_()])))
			- kw2_[k]->value(T) * macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_()])))
			* R.d_slip_d_s(g, i, stress, Q, history, L, T, fixed) * copysign(1.0, slip);
	  res.get<Symmetric>(varnames_[k + nadi_() + islip_()]) = 
			(ki1_[k]->value(T) * std::sqrt(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_() + islip_()])))
			- ki2_[k]->value(T) * macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_() + islip_()])))
			* R.d_slip_d_s(g, i, stress, Q, history, L, T, fixed) * copysign(1.0, slip);
    }
  }
  return res;
}

History AMModel::d_hist_d_h(const Symmetric & stress, 
                                const Orientation & Q, 
                                const History & history, 
                                Lattice & L,
                                double T, const SlipRule & R, 
                                const History & fixed) const
{
  consistency(L); 
  auto res = blank_hist().derivative<History>();

  double ddot = k0_ * std::exp(-Q_ / (R_ * T))
	* history.get<double>(varnames_[0])
	* std::exp(-history.get<double>(varnames_[0])/dc_);

  res.get<double>(varnames_[0] + "_" + varnames_[0]) =
	k0_ * std::exp(-Q_ / (R_ * T))
	* std::exp(-history.get<double>(varnames_[0]) / dc_)
	* (1 - history.get<double>(varnames_[0]) / dc_);

  for (size_t j = 0; j < size(); j++) {
	std::string other = varnames_[j];
	res.get<double>(varnames_[0] + "_" + other) += 0.0;
  }	

  for (size_t g = 0; g < L.ngroup(); g++) {
    for (size_t i = 0; i < L.nslip(g); i++) {
      size_t k = L.flat(g,i);
	  History dslip = R.d_slip_d_h(g, i, stress, Q, history, L, T, fixed);
	  double slip = R.slip(g, i, stress, Q, history, L, T, fixed);
	  res.get<double>(varnames_[k + nadi_()] + "_" + varnames_[k + nadi_()]) = 
		(0.5 * kw1_[k]->value(T)
		* std::pow(macaulay(history.get<double>(varnames_[k + nadi_()])), -0.5)
		- kw2_[k]->value(T)) * std::fabs(slip)
		- 3 / lambda_ * std::pow(macaulay(history.get<double>(varnames_[k + nadi_()])), 0.5)
		* ddot * T / (T + Tr_) * ftr_;
		
	  res.get<double>(varnames_[k + nadi_() + islip_()] + "_" + varnames_[k + nadi_() + islip_()]) = 
		(0.5 * ki1_[k]->value(T)
		* std::pow(macaulay(history.get<double>(varnames_[k + nadi_() + islip_()])), -0.5)
		- ki2_[k]->value(T)) * std::fabs(slip);

	  
	  for (size_t j = 0; j < size(); j++) {
		std::string other = varnames_[j];
		if (j < nadi_()) {
		  res.get<double>(varnames_[k + nadi_()] + "_" + other) += 
			(kw1_[k]->value(T) * std::sqrt(macaulay(history.get<double>(varnames_[k + nadi_()])))
			- kw2_[k]->value(T) * macaulay(history.get<double>(varnames_[k + nadi_()])))
			* dslip.get<double>(other) * copysign(1.0, slip)
			- 2 / lambda_ * std::pow(macaulay(history.get<double>(varnames_[k + nadi_()])), 1.5)
			* k0_ * std::exp(-Q_ / (R_ * T)) * T / (T + Tr_) * ftr_
			* std::exp(-history.get<double>(varnames_[0]) / dc_)
			* (1 - history.get<double>(varnames_[0]) / dc_);
			
		  res.get<double>(varnames_[k + nadi_() + islip_()] + "_" + other) += 
			(ki1_[k]->value(T) * std::sqrt(macaulay(history.get<double>(varnames_[k + nadi_() + islip_()])))
			- ki2_[k]->value(T) * macaulay(history.get<double>(varnames_[k + nadi_() + islip_()])))
			* dslip.get<double>(other) * copysign(1.0, slip)
			+ 2 / lambda_ * std::pow(macaulay(history.get<double>(varnames_[k + nadi_()])), 1.5)
			* k0_ * std::exp(-Q_ / (R_ * T)) * T / (T + Tr_) * ftr_
			* std::exp(-history.get<double>(varnames_[0]) / dc_)
			* (1 - history.get<double>(varnames_[0]) / dc_);
			
		}
		else if (j >= nadi_() and j < wslip_() + nadi_()){
		  res.get<double>(varnames_[k + nadi_()] + "_" + other) += 
			(kw1_[k]->value(T) * std::sqrt(macaulay(history.get<double>(varnames_[k + nadi_()])))
			- kw2_[k]->value(T) * macaulay(history.get<double>(varnames_[k + nadi_()])))
			* dslip.get<double>(other) * copysign(1.0, slip);
			
		  res.get<double>(varnames_[k + nadi_() + islip_()] + "_" + other) += 
			(ki1_[k]->value(T) * std::sqrt(macaulay(history.get<double>(varnames_[k + nadi_() + islip_()])))
			- ki2_[k]->value(T) * macaulay(history.get<double>(varnames_[k + nadi_() + islip_()])))
			* dslip.get<double>(other) * copysign(1.0, slip);  
			
		  if (k + nadi_() == j){
			res.get<double>(varnames_[k + nadi_() + islip_()] + "_" + other) += 
			  3 / lambda_ * std::sqrt(macaulay(history.get<double>(varnames_[k + nadi_()])))
			  * ddot * T / (T + Tr_) * ftr_;
		  } 
		}
		else {
		  res.get<double>(varnames_[k + nadi_()] + "_" + other) +=
			(kw1_[k]->value(T) * std::sqrt(macaulay(history.get<double>(varnames_[k + nadi_()])))
			- kw2_[k]->value(T) * macaulay(history.get<double>(varnames_[k + nadi_()])))
			* dslip.get<double>(other) * copysign(1.0, slip);
			
		  res.get<double>(varnames_[k + nadi_() + islip_()] + "_" + other) +=
			(ki1_[k]->value(T) * std::sqrt(macaulay(history.get<double>(varnames_[k + nadi_() + islip_()])))
			- ki2_[k]->value(T) * macaulay(history.get<double>(varnames_[k + nadi_() + islip_()])))
			* dslip.get<double>(other) * copysign(1.0, slip);
		}
	  }
    }
  }
  return res;
}

History AMModel::d_hist_d_h_ext(const Symmetric & stress, 
                                    const Orientation & Q,
                                    const History & history,
                                    Lattice & L, double T, const SlipRule & R,
                                    const History & fixed, 
                                    std::vector<std::string> ext) const
{
  consistency(L);
  History res = blank_hist().history_derivative(history.subset(ext)).zero();

  for (auto vn : ext) {
    res.get<double>(varnames_[0] + "_" + vn) = 0;
  }	

  for (size_t g = 0; g < L.ngroup(); g++) {
    for (size_t i = 0; i < L.nslip(g); i++) {
      Lattice::SlipType stype = L.slip_type(g,i); 
      size_t k = L.flat(g,i);
      if (stype == Lattice::SlipType::Slip) {
        History dslip = R.d_slip_d_h(g, i, stress, Q, history, L, T, fixed);
        for (auto vn : ext) {
          res.get<double>(varnames_[k + nadi_()] + "_" + vn) = 
			(kw1_[k]->value(T) * std::sqrt(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_()])))
			- kw2_[k]->value(T) * macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_()])))
			* dslip.get<double>(vn);
		  res.get<double>(varnames_[k + nadi_() + islip_()] + "_" + vn) = 
			(ki1_[k]->value(T) * std::sqrt(macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_() + islip_()])))
			- ki2_[k]->value(T) * macaulay(history.get<double>(varnames_[L.flat(g,i) + nadi_() + islip_()])))
			* dslip.get<double>(vn);
        }				
      }

    }
  }
  return res;
}

void AMModel::consistency(Lattice & L) const
{
  if (L.ntotal() + L.ntotal() + nadi_() != size()) {
    throw std::logic_error("Lattice and hardening matrix sizes do not match");
  }
}

}