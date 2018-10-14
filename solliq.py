
# coding: utf-8

# ## solliq - Solidus and liquidus parameterizations for geomaterials
# ####  Thomas Ruedas (Thomas dot Ruedas at dlr dot de) - solliq v.1.3, 14/10/2018
# The solidi and liquidi of geomaterials are phase boundaries and depend on pressure $p$, temperature $T$, and composition. Solidi and liquidi of peridotite and basalt/eclogite (both anhydrous) as well as iron and iron-sulfur alloys are given here as parameterizations in the form $T(p)$, possibly with additional terms that account for compositional variations. $T$ is always given in K, $p$ in GPa.
# 
# #### Note on citation
# Some of the parameterizations implemented here are published in the scientific literature:
# - [Terrestrial peridotite upper-mantle solidus](#Tsol_E): Hirschmann (2000), Hirschmann et al. (2009), with the modifications described in the corresponding section below
# - [Martian peridotite solidus](#Tsol_M): Ruedas & Breuer (2017)
# - [CMAS solidus](#Tsol_CMAS): Ruedas & Breuer (2018)
# - [Fe and alkali corrections](#Fealk): Ruedas & Breuer (2018)
# - [Chondritic lower-mantle solidus](#Tsol_chon): Andrault et al. (2018)
# 
# If you use this software for your work, please acknowledge it and also cite the corresponding literature where applicable.
# 
# This software is released under the GNU General Public License v.3.
# 
# #### Contents
# - [Fitting approach](#fit)
# - [Peridotite solidus](#Tsol_per)
#   - [Terrestrial peridotite](#Tsol_E)
#   - [CMAS solidus](#Tsol_CMAS)
#   - [Martian solidus](#Tsol_M)
#   - [Fe and alkali corrections](#Fealk)
#   - [Chondritic solidus](#Tsol_chon)
# - [Peridotite liquidus](#Tliq_per)
# - [Basalt/eclogite solidus](#Tsol_bas)
# - [Basalt/eclogite liquidus](#Tliq_bas)
# - [Iron and binary iron alloys](#alloy)
#   - [Iron](#Fe)
#   - [FeS](#FeS)
#   - [Eutectic](#eutect)
#   - [Interpolated melting curves](#intpol)
# - [Main program (user interface)](#main)
# - [Version history](#hist)
# - [References](#ref)

# ### <a id='fit'></a>Fitting approach
# The parameterizations of the melting curves are constructed by performing least-squares fits of the available experimental data to empirical model functions, in particular polynomials of first to third degree or to the Simon-Glatzel equation (Simon & Glatzel, 1929), which is usually given as
# \begin{equation}
# T_\mathrm{m}(p)=T_\mathrm{ref}\left(\frac{p-p_\mathrm{ref}}{a'}+1\right)^\frac{1}{c'},
# \end{equation}
# where $p_\mathrm{ref}$ and $T_\mathrm{ref}$ are a reference pressure and temperature often chosen to correspond to the melting point at surface (zero) pressure. The equation is used here in the modified form
# \begin{equation}
# T_\mathrm{m}(p)=\frac{T_\mathrm{ref}}{a'^\frac{1}{c'}}(p-p_\mathrm{ref}+a')^\frac{1}{c'}=a(p+b)^c,
# \end{equation}
# with $a=T_\mathrm{ref}/a'^\frac{1}{c'}$, $b=a'-p_\mathrm{ref}$, and $c=\frac{1}{c'}$. For each dataset, several model functions were tried, and the one with the least rms residual was usually chosen if it met certain requirements. The most important requirements reflect experimental experience and are that the curve should be monotonically increasing and have no inflexion points in the $p$ range of interest and that its slope should decrease with increasing $p$. The functions are generally not meant to be used for extrapolation much beyond the range of data coverage.
# 
# If the data warranted it, the possible effects of phase transitions of the solid phase were taken into account by carrying out piecewise fits; such solid-solid phase transitions may manifest themselves for instance in the form of kinks in the melting curve, although sharp kinks are expected in chemically pure systems rather than solid-solution systems such as several of those of interest here. In such piecewise fits, continuity conditions were imposed on solidus fits to data from natural compositions that included solid solutions where necessary; the practice to allow jumps in the solidus followed in older versions is abandoned now, because smooth transitions are expected in all systems that feature solid solutions. The most important example is the transition from a terrestrial upper-mantle mineral assemblage to a lower-mantle assemblage.

# ### <a id='Tsol_per'></a>Peridotite solidus
# #### <a id='Tsol_E'></a>Terrestrial peridotite
# Data for terrestrial peridotite are available for the entire depth range of the Earth's mantle. The following parameterization is based on Hirschmann (2000, tab.2 and pers.comm.) and Hirschmann et al. (2009) for the upper mantle, and the downward shift by 35 K suggested by Katz et al. (2003) is applied. In the lower mantle, the linear Hirschmann et al. fit is replaced with a curved model function fitted to some of the data also used by Hirschmann et al. (2009), supplemented by additional data from Hirose & Fei (2002), Fiquet et al. (2010), and Zerr et al. (1997,1998), covering 22.5 GPa $< p <$ 122 GPa.
# \begin{equation}
# T_\mathrm{s,E}(p)=
# \begin{cases}
# 1358.81061+132.899012p-5.1404654p^2&p<10\,\mathrm{GPa}\\
# -1.092(p-10)^2+32.39(p-10)+2173.15&10\,\mathrm{GPa}\leq p < 23.5\,\mathrm{GPa}\\
# -1647.15-5.85p+1329.126\ln(p)&\text{else.}
# \end{cases}
# \end{equation}
# A Simon-Glatzel-type fit could not be carried out for the lower mantle data with the constraint to match the Hirschmann et al. (2009) curve at 23.5 GPa. The high-$p$ study on a peridotitic composition by Nomura et al. (2014) was excluded, because they used a synthetic composition (pyrolite) and arrived at a substantially lower solidus; futhermore, their data have strong scatter and provide only rather wide brackets on the solidus temperature.
# 
# The lower-mantle solidus permits plausible extrapolation to higher $p$, e.g., for application in exoplanets, but it must be kept in mind that a transition of bridgmanite to postperovskite is expected somewhere between 130 and 170 GPa, and there are no experimental constraints on the solidus in a postperovskite lithology.

# In[ ]:


import numpy as np
def Tsol_Earth(p):
    if p < 10:
        Tsol=1358.81061+(132.899012-5.1404654*p)*p # w/ Katz shift
    elif p >= 10 and p < 23.5:
        pr=p-10.
        Tsol=(-1.092*pr+32.39)*pr+2173.15 # w/ Katz shift
    else:
        Tsol=-1647.15020384-5.85124635*p+1329.1263647*np.log(p)
    return Tsol


# #### <a id='Tsol_CMAS'></a>CMAS solidus
# The CaO-MgO-Al<sub>2</sub>O<sub>3</sub>-SiO<sub>2</sub> (CMAS) system is the simplest oxide system capable of reproducing all four phases of a fertile lherzolite, i.e., olivine (forsterite Mg<sub>2</sub>SiO<sub>4</sub>), orthopyroxene (enstatite MgSiO<sub>3</sub>+wollastonite CaSiO<sub>3</sub>), clinopyroxene (diopside CaMgSi<sub>2</sub>O<sub>6</sub>), and either plagioclase (anorthite CaAl<sub>2</sub>Si<sub>2</sub>O<sub>8</sub>), spinel (MgAl<sub>2</sub>O<sub>4</sub>), or garnet (pyrope Mg<sub>3</sub>Al<sub>2</sub>Si<sub>3</sub>O<sub>12</sub>+grossular Ca<sub>3</sub>Al<sub>2</sub>Si<sub>3</sub>O<sub>12</sub>). It can also serve as a proxy for the presumably very Fe-poor peridotite of Mercury. Data are available for $p$ up to 21.5 GPa (Asahara et al., 1998; Asahara & Ohtani, 2001; Gudfinnsson & Presnall, 1996, 2000; Herzberg et al., 1990; Klemme & O'Neill, 2000; Litasov & Ohtani, 2002; Liu & O'Neill, 2004; Milholland & Presnall, 1998; Presnall et al., 1979) and have been fitted with a uniform polynomial, i.e., without considering phase transitions that may introduce kinks (which would probably be rather sharp in this case due to the absence of solid solutions in some of the phases):
# \begin{equation}
# T_\mathrm{s,CMAS}(p)=1477.54+139.391p-7.7545p^2+0.160258p^3
# \end{equation}
# (Ruedas & Breuer, 2018, App.B). For $p>23.5$ GPa, which would correspond to the lower mantle, we use the terrestrial peridotite solidus, tentatively shifted up by about 139 K to ensure continuity, but there are no experimental constraints.

# In[ ]:


def Tsol_CMAS(p):
    if p < 23.5:
        Tsol=1477.54+(139.391-(7.7545-0.160258*p)*p)*p
    else:
        Tsol=Tsol_Earth(p)+139.2162
    return Tsol


# #### <a id='Tsol_M'></a>Martian solidus
# The peridotite of the martian mantle has a different composition than its terrestrial counterpart, with the substantially lower Mg# being the most conspicuous difference. Ruedas & Breuer (2017, App. A) fitted experiments on martian model compositions (Bertka & Holloway, 1994; Collinet et al., 2015; Matsukage et al., 2013; Schmerr et al., 2001) to polynomials to produce the parameterization
# \begin{equation}
# T_\mathrm{s,M}(p)=
# \begin{cases}
# 0.118912p^3-6.37695p^2+130.33p+1340.38&\text{for $p<23$ GPa}\\
# 62.5p+975&\mathrm{else.}
# \end{cases}
# \end{equation}
# With the exception of the Collinet et al. data, all data used for the fit have been shifted downward by 30 K to correct for the absence of K in the sample, using an estimate on the effect of K on the solidus derived from Wang & Takahashi (2000). As Collinet et al. did not provide subsolidus data, the melting degree-temperature relations from their experiments were extrapolated linearly at each pressure, and the resulting solidus estimate was weighted doubly. Note that the function for $p>23$ GPa is constrained by only four points and only to 25 GPa, and so extrapolation to much higher $p$ is strongly discouraged. For Mars itself, this is not a problem, because Mars is not expected to have a perovskitic basal layer, but it would be a concern for exoplanets that are larger than Mars and have an iron-rich mantle.

# In[ ]:


def Tsol_Mars(p):
    if p < 23:
        Tsol=((0.118912*p-6.37695)*p+130.33)*p+1340.38
    else:
        Tsol=62.5*p+975
    return Tsol


# #### <a id='Fealk'></a>Fe and alkali corrections
# For peridotite with compositions not covered by experimental data, we can try to interpolate between better constrained solidi. The reference is the terrestrial solidus, and interpolation is carried out by linear interpolation based on Mg# between it and either the martian or the CMAS solidus corrected for the difference with Earth in their alkali (Na+K) contents. For these corrections, the oxide composition (specifically, the MgO, FeO, Na<sub>2</sub>O, and K<sub>2</sub>O content) of the target material needs to be known; the terrestrial and martian reference oxide compositions are taken from Palme & O'Neill (2014, Tabs. 3,4) and Taylor & McLennan (2009, Table 5.4), respectively. An alkali effect $\mathrm{d}T_\mathrm{s}/\mathrm{d}X_\mathrm{Na+K}=-14951.52$ K has been estimated from values from Hirschmann (2000) and Wang & Takahashi (2000). The interpolated solidus is then
# \begin{equation}
# T_\mathrm{s}(p)=(1-w_\mathrm{Mg\#})T_\mathrm{s,E}+w_\mathrm{Mg\#}T_\mathrm{sr}+
# \frac{\mathrm{d}T_\mathrm{s}}{\mathrm{d}X_\mathrm{Na+K}}(X_\mathrm{Na+K}-X_\mathrm{Na+K,E}),
# \end{equation}
# where
# \begin{align}
# w_\mathrm{Mg\#}&=\frac{\mathrm{Mg\#}_\mathrm{E}-\mathrm{Mg\#}}{\mathrm{Mg\#}_\mathrm{E}-\mathrm{Mg\#}_2}\\
# T_\mathrm{sr}&=T_\mathrm{s,2}-\frac{\mathrm{d}T_\mathrm{s}}{\mathrm{d}X_\mathrm{Na+K}}(X_\mathrm{Na+K,2}-X_\mathrm{Na+K,E}).
# \end{align}
# The subscripts E and 2 indicate the Earth and the other reference system (i.e., Mars for $\mathrm{Mg\#}<\mathrm{Mg\#}_\mathrm{E}$ and CMAS otherwise), respectively. $T_\mathrm{sr}$ is hence the solidus of the other reference system with the alkali effect reduced to that of the Earth.

# In[ ]:


# oxide contents of reference bulk silicate planet compositions
ox_E={'MgO': 36.77e-2,'FeO': 8.1e-2,'Na2O': 3.5e-3,'K2O': 3e-4} # Earth
ox_M={'MgO': 30.2e-2,'FeO': 17.9e-2,'Na2O': 5e-3,'K2O': 4e-4} # Mars
def Tsol_intp(p,ox):
    global Mg0,Mg0_E
    uMgO=u['Mg']+u['O']
    uFeO=u['Fe']+u['O']
    el1=ox_E['MgO']/uMgO
    el2=ox_E['FeO']/uFeO
    Mg0_E=el1/(el1+el2)
    xwNaK_E=ox_E['Na2O']+ox_E['K2O']
    dTsdNaK=-14951.52  # K per mass frac.
#   compositional variables
    el1=ox['MgO']/uMgO
    el2=ox['FeO']/uFeO
    Mg0=el1/(el1+el2)
#   interpolation
    TsE=Tsol_Earth(p)
    if Mg0 < Mg0_E:
#       iron-rich, interpolate between Earth and Mars solidi
        el1=ox_M['MgO']/uMgO
        el2=ox_M['FeO']/uFeO
        w_dMg0=(Mg0_E-Mg0)/(Mg0_E-el1/(el1+el2))
        TsM=Tsol_Mars(p)-dTsdNaK*(ox_M['Na2O']+ox_M['K2O']-xwNaK_E)
    else:
#       iron-poor, interpolate between Earth and CMAS solidi
        w_dMg0=(Mg0-Mg0_E)/(1-Mg0_E)
        TsM=Tsol_CMAS(p)+dTsdNaK*xwNaK_E
#   weighted linear interpolation plus alkali correction relative to terrestrial
    return (1.-w_dMg0)*TsE+w_dMg0*TsM+dTsdNaK*(ox['Na2O']+ox['K2O']-xwNaK_E)


# #### <a id='Tsol_chon'></a>Chondritic solidus
# Some melting experiments have also been conducted on various natural and synthetic "chondritic" compositions. The fit implemented here follows Andrault et al. (2018) in treating the data for the (terrestrial) upper and lower mantle pressure range separately. In the latter, only the data from Andrault et al. (2011, synthetic CMASF) are available, as a single additional point from Ohtani & Sawamoto (1987) used a simplified (CMASF) composition and is probably not reliable. For lower pressures, we supplement the extensive dataset from Andrault et al. (2018, synthetic CMASFNCrT) with some points from Berthet et al. (2009, Indarch EH4), Ohtani (1987, synthetic CMASF), and Takahashi (1983, Yamato Y-74191 L3 chondrite) that seem to fit in well. Additional points from Agee & Draper (2004, natural and synthetic Homestead) and Cartier et al. (2014, Hvittis enstatitic chondrite EL6) were too far above the general trend to be considered. Data for carbonaceous chondrites were excluded. In summary, the implemented solidus is
# \begin{equation}
# T_\mathrm{s,ch}(p)=
# \begin{cases}
# 1337.159(p+0.9717)^{0.1663}&\text{for $p<27.5$ GPa}\\
# 520.735(p+9.6535)^{0.4155}&\mathrm{else,}
# \end{cases}
# \end{equation}
# whereby the latter is simply a rewritten form of the lower-mantle fit given by Andrault et al. (2018). This solidus has data coverage up to 140 GPa.

# In[ ]:


def Tsol_chon(p):
    if p < 27.5:
        Tsol=1337.159*(p+0.9717)**0.1663
    else:
        Tsol=520.735*(p+9.6535)**0.4155
    return Tsol


# ### <a id='Tliq_per'></a>Peridotite liquidus
# For the liquidus, a distinction should be made between a system that retains all of the melt produced and thus has a constant bulk composition (batch melting) and a system in which the produced melt is removed immediately, leaving behind a solid system with a different bulk composition that changes continuously as melting progresses (fractional melting). For many applications in mantle dynamics, the latter case is considered more appropriate and implies, at least in a simplified treatment, that the liquidus to be used is that of the last phase to remain solid as the melting degree increases (cf. Iwamori et al., 1995, p.257). This "liquidus phase" varies with pressure. Based on the phase diagram for peridotite by Litasov & Ohtani (2007), the function switches through the succession forsterite-pyrope-periclase as $p$ increases; this phase diagram is derived from batch melting experiments, but due to lack of experimental work representing fractional melting, it is assumed that the same liquidus phases occur in fractionally melting mantle in approximately the same $p$ ranges. The switch to periclase is triggered by `wpc`, which represents the mass fraction of periclase, becoming non-zero; this is currently hard-wired to occur linearly between 23 and 24 GPa. The individual liquidus curves are a polynomial parameterization derived from experimental data for forsterite (Davis & England, 1964; Navrotsky et al., 1989; Ohtani & Kumazawa, 1981; Presnall & Walter, 1993; Richet et al., 1993) and Simon-Glatzel-type fits to experimental data for pyrope (Boyd & England, 1962; Irifune & Ohtani, 1986; Kudo & Ito, 1996; Ohtani et al., 1981; Shen & Lazor, 1995; Zhang & Herzberg, 1994) and to first-principles simulations of periclase by Alfè (2005), as no usable experimental data were available for the latter; experimentally determined melting curves of periclase still have large uncertainty and may vary by hundreds or even more than 1000 K at high $p$.
# \begin{equation}
# T_\mathrm{l,per}=
# \begin{cases}
# 2160.6+64.7109p-3.97463p^2+0.0957894p^3&p<13.244\,\text{GPa (forsterite)}\\
# 589.691(p+11.9076)^{0.453188}&p\geq 13.244\,\text{GPa and no periclase present (pyrope)}\\
# 1705.67(p+5.20372)^{0.315027}&\text{(periclase)}
# \end{cases}
# \end{equation}
# The melting curve of pyrope could in theory be extrapolated to normal pressure, but nominal melting temperature values at normal pressure have not been used for fitting, because pyrope melts incongruently under these conditions.

# In[ ]:


def Tliq_per(p):
    pxfopy=13.244 # GPa, transition p from fo to py
    if p > pxfopy:
        wpc=min(max(0.,p-23),1.)
        if wpc == 1:
            Tliq_per=Tliqpc(p)
        elif wpc > 0:
            Tliq_per=(1.-wpc)*Tliqpy(p)+wpc*Tliqpc(p)
        else:
            Tliq_per=Tliqpy(p)
    else:
        Tliq_per=Tliqfo(p)
    return Tliq_per
# forsterite melting curve
def Tliqfo(p):
    pmaxfit=14.64
    if p > pmaxfit:
        print("WARNING: p>%.2f GPa, Tliq (forsterite) assumed const." % pmaxfit)
        Tliqfo=2557.6807
    else:
        Tliqfo=2160.6+(64.7109-(3.97463-0.0957894*p)*p)*p
    return Tliqfo
# pyrope melting curve
def Tliqpy(p):
    return 589.691*(p+11.9076)**0.453188
# periclase melting curve
def Tliqpc(p):
    return 1705.67*(p+5.20372)**0.315027


# For possible future use with peridotite or basalt/eclogite, a melting curve for bridgmanite was also constructed from experimental data (Zerr & Boehler, 1993):
# \begin{equation}
# T_\mathrm{l,br}(p)=-0.338678p^2+79.9414p+1169.02.
# \end{equation}
# It is not being used in the current version.

# In[ ]:


# bridgmanite melting curve
def Tliqbr(p):
    return (-0.338678*p+79.9414)*p+1169.02


# For completeness and because it is frequently used, we also construct a "batch melting liquidus" for peridotite from experimental data on natural terrestrial compositions (Fiquet et al., 2010, KLB-1; Herzberg et al., 1990, KLB-1; Ito & Kennedy, 1967, KA 64-16; Ito & Takahashi, 1987, PHN1611; Scarfe & Takahashi, 1986, PHN1611; Takahashi, 1986, KLB-1; Takahashi & Scarfe, 1985, KLB-1; Takahashi et al., 1993, KLB-1; Trønnes & Frost, 2002, pyrolite/KLB-1):
# \begin{equation}
# T_\mathrm{lb,per}=
# \begin{cases}
# 2067.59+25.1327p-0.2455p^2&p<22.76\,\text{GPa}\\
# 2207.71(p-20.7372)^{0.182}&\text{else.}
# \end{cases}
# \end{equation}
# Some other datasets from the literature were not used, because they use synthetic, simplified or otherwise deviating compositions.

# In[ ]:


def Tliqb_per(p):
    if p < 22.76:
        Tm=2067.59+(25.1327-0.2455*p)*p
    else:
        Tm=2207.71*(p-20.7372)**0.182
    return Tm


# For a chondritic composition, we use the datasets by Agee & Draper (2004, natural and synthetic Homestead), Andrault et al. (2011, synthetic CMASF), Berthet et al. (2009, Indarch EH4), Ohtani (1987, synthetic CMASF), and Takahashi (1983, Yamato Y-74191 L3 chondrite); again, carbonaceous chondrites were avoided. As the data do not warrant making a distinction between the upper and lower mantle pressure range, we fit the whole dataset with a Simon-Glatzel-type equation:
# \begin{equation}
# T_\mathrm{lb,ch}=310.317(p+29.102)^{0.5337}.
# \end{equation}

# In[ ]:


def Tliqb_chon(p):
    return 310.317*(p+29.102)**0.5337


# ### <a id='Tsol_bas'></a>Basalt/eclogite solidus
# Basalt consists mostly of clinopyroxene and plagioclase. When the plagioclase transforms into garnet, the basalt becomes eclogite. Data for the solidus of basalt are more scarce, and so we interpolate between a few points measured from the (terrestrial) MORB basalt/eclogite phase diagram provided by Litasov & Ohtani (2007) for the stability field of basalt, which we extend to 3 GPa in order to achieve continuity with the eclogite parameterization, for which experimental data were fitted. When the eclogite transforms into a bridgmanite-bearing high-$p$ assemblage, the function shifts linearly to the high-$p$ melting curve over a $p$ interval from 26 to 30 GPa. The bound of 26 GPa separates the subsets of experimental data used for fitting the eclogite and the high-$p$ solidi and should therefore not be changed; the 30 GPa bound is somewhat arbitrary and may be changed if desired. The experimental data are from Andrault et al. (2014), Hirose et al. (1999), Hirose & Fei (2002), Pradhan et al. (2015), and Yasuda et al. (1994); from the Andrault et al. data, only those above ~56 GPa are used, because the lower-$p$ data lie significantly beneath those of the other studies.
# \begin{equation}
# T_\mathrm{s,bas}(p)=
# \begin{cases}
# 1.83261p^4+13.0994p^2+1340&p<3\,\text{GPa}\\
# -1.273p^2+78.676p+1380.92&3\,\text{GPa}\leq p<26\,\text{GPa}\\
# -0.0503p^2+19.277p+2165.09&p\geq 30\,\text{GPa.}
# \end{cases}
# \end{equation}

# In[ ]:


def Tsol_bas(p):
    pgbu=30. # upper boundary of transition
    Tsgt=lambda p: (-1.273*p+78.676)*p+1380.92
    Tsbr=lambda p: (-0.0503*p+19.277)*p+2165.09
    if p < 3:
#       plagioclase stability field
        p2=p*p
        Ts=(1.83261*p2+13.0994)*p2+1340.
    elif p >= 3 and p < 26:
#       garnet/majorite stability field
        Ts=Tsgt(p)
    elif p >= 26 and p < pgbu:
#       transition: linear shift from Tsgt to Tsbr
        Ts=(Tsgt(p)*(pgbu-p)+Tsbr(p)*(p-26))/(pgbu-26)
    else:
#       perovskite stability field
        Ts=Tsbr(p)
    return Ts


# ### <a id='Tliq_bas'></a>Basalt/eclogite liquidus
# According to the phase diagram from Litasov & Ohtani (2007), the liquidus phase in basalt melting also changes with $p$. We switch between liquidus phases as $p$ increases and use again parameterized melting curves from experimental data, this time for forsterite as above, for diopside (Boyd & England, 1963; Gasparik, 1996; Richet & Bottinga, 1984; Shen & Lazor, 1995; Yoder, 1952), for pyrope as above, and for Ca-perovskite (Gasparik et al., 1994; Zerr et al., 1997).
# \begin{equation}
# T_\mathrm{l,bas}(p)=
# \begin{cases}
# 2160.6+64.7109p-3.97463p^2+0.0957894p^3&p<1\,\text{GPa (forsterite)}\\
# 1665.25+133.484p-9.02117p^2+0.243279p^3&1\,\mathrm{GPa}\leq p < 3.728\,\text{GPa (diopside)}\\
# 589.691(p+11.9076)^{0.453188}&3.728\,\mathrm{GPa}\leq p < 24\,\text{GPa (pyrope)}\\
# 394.881(p+11.1079)^{0.593651}&\text{else (Ca-perovskite).}
# \end{cases}
# \end{equation}
# This parameterization has some particularly large steps, e.g. at 1 GPa, which imply strong changes in the width of the partial melting interval and correspondingly strong variations with isobaric melt productivity in different stability fields.

# In[ ]:


def Tliq_bas(p):
    if p < 1:
        Tliq=Tliqfo(p)
    elif p < 3.728:
        Tliq=Tliqdi(p)
    elif p < 24:
        Tliq=Tliqpy(p)
    else:
        Tliq=TliqCapv(p)
    return Tliq
# diopside melting curve
def Tliqdi(p):
    return 1665.25+(133.484-(9.02117-0.243279*p)*p)*p
# Ca-perovskite melting curve
def TliqCapv(p):
    return 394.881*(p+11.1079)**0.593651


# The phase diagram from Litasov & Ohtani (2007) and some experimental data from Hirose & Fei (2002) and Yasuda et al. (1994) are also used to construct a "batch melting liquidus" for basalt in a way similar to the solidus; for $p\gtrsim 22$ GPa, however, the melting curve of Ca-perovskite is used as eclogite transforms into a perovskitic assemblage and Ca-perovskite has been found to be the likely liquidus phase up to pressures corresponding to the terrestrial CMB:
# \begin{equation}
# T_\mathrm{lb,bas}(p)=
# \begin{cases}
# 21.5909p+1488&p<0.88\,\mathrm{GPa}\\
# -2.24727p^2+97.6753p+1422.79&0.88\,\mathrm{GPa}\leq p<3.12\,\mathrm{GPa}\\
# 1581.57(p-1.468)^{0.151}&3.12\,\mathrm{GPa}\leq p<22\,\mathrm{GPa}\\
# T_\mathrm{l,Capv}(p)&\text{else.}
# \end{cases}
# \end{equation}
# The interval between 0.88 and 3.12 GPa is derived from Litasov & Ohtani (2007) and serves as a link between the lowest-$p$ segment and the range constrained by experimental data. Note, however, that around 22 GPa this liquidus gets very close to the solidus and that it is not at all constrained at higher $p$; its use at $p\gtrsim 20$ GPa should probably be avoided.

# In[ ]:


def Tliqb_bas(p):
    if p < 0.88:
#       plagioclase stability field
        Tliq=21.5909*p+1488.
    elif p < 3.12:
#       garnet stability field up to intersection with data fit
        Tliq=(-2.24727*p+97.6753)*p+1422.79
    elif p < 22:
#       approx. garnet stability field, from exp. data
        Tliq=1581.57*(p-1.468)**0.151
    else:
#       Ca-perovskite stability field
        Tliq=TliqCapv(p)
    return Tliq


# ### <a id='alloy'></a>Iron and binary iron alloys
# #### <a id='Fe'></a>Iron
# With increasing pressure, different Fe allotropes are on the solid side of the melting curve: body-centered cubic (bcc) $\delta$-Fe at low $p$, face-centered cubic (fcc) $\gamma$-Fe at intermediate $p$, and hexagonal (hcp) $\epsilon$-Fe at high $p$. However, the fcc-hcp-liquid triple point is somewhat uncertain, and the melting curve especially for hcp-Fe is still controversial, as there are two quite different trends to be found in published experimental data at $p\gtrsim 30$ GPa. For this reason, we consider two alternative melting curves, one following a fairly flat $p$ gradient in the fcc stability field and another following a much steeper gradient. At this point, the even more controversial high-$p$ body-centered phase hypothesized by some authors to replace hcp-Fe above a certain high pressure is not included, and hcp-Fe is assumed to be stable to the highest $p$ considered. Experimental melting curve data are from Liu & Bassett (1975), Strong et al. (1973), and Swartzendruber (1982) for $\delta$-Fe, from Ahrens et al. (2002), Anzellini et al. (2013), Aquilanti et al. (2015), Boehler (1993), Boehler & Ross (2007), Boehler et al. (2008), Buono & Walker (2011), Jackson et al. (2013), Liu & Bassett (1975), Ringwood & Hibberson (1990), Shen et al. (1998, 2004), Strong et al. (1973), Williams et al. (1987), and Zhang et al. (2016) for fcc-Fe, and from Anzellini et al. (2013), Aquilanti et al. (2015), Boehler (1993), Boehler & Ross (2007), Boehler et al. (2008), Ma et al. (2004), Murphy et al. (2011), Nguyen & Holmes (2004), Shen et al. (1998), and Tateno et al. (2010) for hcp-Fe.
# 
# In order to construct the "flat" and "steep" fits, the data have been divided into two subsets that have some overlap at $p\lesssim 30$ GPa as well as at $p>300$ GPa, and some points were discarded. In the data subset used for describing the "flat" gradient, it seems useful to subdivide the $p$ into three segments according to the stable solid phase and construct peacewise fits, i.e.,
# \begin{equation}
# T_\mathrm{m,Fe}=
# \begin{cases}
# 1811+30.9724p&p\leq 8.044\,\text{GPa (bcc/$\delta$)}\\
# 1907.47+19.8675p-0.1103p^2&8.044\,\mathrm{GPa}<p\leq 87.06\,\text{GPa (fcc/$\gamma$)}\\
# 2130.55+7.2054p+0.00571047p^2&87.06\,\mathrm{GPa}<p\text{ (hcp/$\epsilon$).}
# \end{cases}
# \end{equation}
# As an exception to the rule, the hcp interval of this fit is slightly bent *upward* and should therefore not be used beyond 400 GPa; excluding the points at $p>300$ GPa from Tateno et al. (2010) from the fit would result in a markedly stronger upward bent, however, and is therefore further from the expected trend. The data subset used for the "steep" gradient has an overall appearance that seems to allow for a uniform fit, ignoring possible kinks at phase boundaries, which cannot be identified with any certainty:
# \begin{equation}
# T_\mathrm{m,Fe}=1811+24.7307p-0.0627041p^2+6.14455\cdot 10^{-5}p^3.
# \end{equation}
# The zero-pressure melting point of $\delta$-Fe has been fixed at 1811 K (Swartzendruber, 1982) in both cases.

# In[ ]:


def Tliq_Fe(p,mode):
    if mode == 'f':
#       flat gradient
        if p <= 8.044:
#           delta phase (bcc)
            Tm=1811.+30.9724*p
        elif p <= 87.06:
#           gamma phase (fcc)
            Tm=1907.47+(19.8675-0.1103*p)*p
        else:
#           epsilon phase (hcp)
            Tm=2130.55+(7.2054+0.00571047*p)*p
    else:
#       steep gradient
        Tm=1811+(24.7307-(0.0627041-6.14455e-5*p)*p)*p
    return Tm


# #### <a id='FeS'></a>FeS
# Melting point data for (stoichometric) FeS are much scarcer. The experimental data from Anderson & Ahrens (1996) and Boehler (1992) are supplemented by some points read from a phase diagram by Ono et al. (2008); additional data from a study by Williams & Jeanloz (1990) were not used, because they show strong scatter and are inconsistent with the other data. The available data suggest that in spite of the occurrence of several phase transitions, a single Simon-Glatzel-type function can be fitted over the entire pressure range up to 360 GPa:
# \begin{equation}
# T_\mathrm{m,FeS}=638.5613(p+13.2844)^{0.3216},
# \end{equation}
# whereby the zero-pressure melting point has been fixed at 1467 K.

# In[ ]:


def Tliq_FeS(p):
    return 638.5613*(p+13.2844)**0.3216


# #### <a id='eutectic'></a>Eutectic
# The melting curve of an Fe-S alloy with a composition between Fe and FeS is not a monotonic function of composition but has a minimum at the intermediate eutectic composition, at which both solid phases and the melt are in equilibrium and which is itself a function of pressure. Experimental work indicates that the molar fraction of S at the eutectic is a monotonically falling function of $p$ and can be represented as
# \begin{equation}
# \bar{X}_\mathrm{eut}(p)=0.925(p+8.444)^{-0.373},
# \end{equation}
# based on data by Brett & Bell (1969), Buono & Walker (2011), Campbell et al. (2007), Chen et al. (2008), Chudinovskikh & Boehler (2007), Fei et al. (1997, 2000), Kamada et al. (2010),  Li et al. (2001), Morard et al. (2007, 2008), Mori et al. (2017), Ryzhenko & Kennedy (1973), Stewart et al. (2007), and Usselman (1975). Campbell et al. (2007) had suggested that $\bar{X}_\mathrm{eut}(p)$ approaches a value of 0.25 at $p\gtrsim 25$ GPa, but as this is in conflict with other data, a value of 0.25 was assumed to apply only to their data up to 36.6 GPa. This fit replaces the exponential fit by Ruedas et al. (2013, App.B), which works only for $p\lesssim 100$ GPa.

# In[ ]:


def Xeut(p):
    return 0.925/(p+8.444)**0.373


# The corresponding temperature $T_\mathrm{eut}$ at the eutectic point is also a function of $p$. We follow Mori et al. (2017) in basing the model on a Simon-Glatzel-type function, but experimental work by Fei et al. (1997, 2000) has shown that the intricacies of the Fe-S phase diagram introduce a minimum somewhere between 10 and 15 GPa, which is here represented by a Gaussian-shaped additional term that is significant only in the pressure range up to $\sim 20$ GPa:
# \begin{gather}
# T_\mathrm{eut}(p)=255(p+8)^{\tau(p)}+600\exp(-0.012(p+2)^2)\\
# \tau(p)=0.49\left\lbrace 1-0.085\left[\frac{1}{2}-\frac{\arctan(0.007(250-p))}{\pi}\right]\right\rbrace;
# \end{gather}
# the monstrosity $\tau(p)$ is needed to reduce the exponent of the first term of $T_\mathrm{eut}$ at high $p$ and introduce a slight, smooth bend between approximately 100 and 150 GPa that ensures that the eutectic does not become greater than the pure-phase liquidi at high $p$. The experimental data for the (in this case visually performed) fit are from the same sources and from Boehler (1996), but in the studies predating Fei et al. (1997), data above 6 GPa had to be dismissed, because they did not account for the aforementioned complexities of the phase diagram under those conditions. As with $\bar{X}_\mathrm{eut}(p)$, this fit replaces the corresponding exponential fit by Ruedas et al. (2013, App.B), which overestimates $T_\mathrm{eut}$ for $p\gtrsim 150$ GPa.

# In[ ]:


def Teut(p):
    from math import exp,atan,pi
    fxp=0.49*(1-0.085*(0.5-atan(0.007*(250-p))/pi))
    return 255*(p+8)**fxp+600*exp(-0.012*(p+2)**2)


# #### <a id='intpol'></a>Interpolated melting curves
# For alloys with an arbitrary binary composition $\bar{X}_\mathrm{S}$, the melting curve of the alloy can be determined from the melting curve of a pure phase. Starting from thermodynamical arguments by Stevenson (1981), Anderson (1998) proposed to determine the solidus depression at a given pressure caused by $N$ solutes in impure Fe by
# \begin{equation}
# \Delta T_\mathrm{m,Fe}(p)=T_\mathrm{m,Fe}(p) \sum\limits_{i=1}^N \ln(1-\bar{X}_i);
# \end{equation}
# in the case considered here, $N=1$ and the solute is S. However, the presence of a eutectic and the existence of FeS as a second endmember limit the applicability to at most the compositions on the iron-rich side of the eutectic. On the sulfur-rich side, one would instead consider the depression of the FeS melting curve by an amount $\bar{X}_\mathrm{Fe+}=\bar{X}_\mathrm{Fe}-\frac{1}{2}$ of Fe in addition to the amount of the stoichometric composition of FeS as a "solute":
# \begin{equation}
# \Delta T_\mathrm{m,FeS}(p)=T_\mathrm{m,FeS}(p) \ln(1-\bar{X}_\mathrm{Fe+}).
# \end{equation}
# In terms of the total amount of sulfur, $\bar{X}_\mathrm{Fe+}=1-\bar{X}_\mathrm{S}-\frac{1}{2}$, and substituting we have
# \begin{equation}
# \Delta T_\mathrm{m,FeS}(p)=T_\mathrm{m,FeS}(p) \ln\left[1-\left(\frac{1}{2}-\bar{X}_\mathrm{S}\right)\right]=
# T_\mathrm{m,FeS}(p) \ln\left(\frac{1}{2}+\bar{X}_\mathrm{S}\right).
# \end{equation}
# Ideally, these expressions would hold on their respective sides of the eutectic up to the eutectic itself, which would then be defined by their intersection, but it turns out that the experimentally determined eutectic runs substantially closer to the pure Fe side of the Fe-FeS join at most $p$.
# 
# Only imposing the experimentally determined $\bar{X}_\mathrm{eut}(p)$ as the boundary between the two domains would violate the condition that the depressed solidi on both sides of the eutectic do not have discontinuities but must be identical at the eutectic itself. Hence we rescale the $\Delta T_\mathrm{m,Fe}$ and $\Delta T_\mathrm{m,FeS}$ using the experimentally constrained $\bar{X}_\mathrm{eut}(p)$ and $T_\mathrm{eut}(p)$ in order to enforce continuity and a correct position of the eutectic (Ruedas et al., 2013):
# \begin{equation}
# T_\mathrm{m}(p,\bar{X}_\mathrm{S})=
# \begin{cases}
# T_\mathrm{m,Fe}(p)-\frac{\ln(1-\bar{X}_\mathrm{S})}{\ln(1-\bar{X}_\mathrm{eut})}(T_\mathrm{m,Fe}(p)-T_\mathrm{eut}(p)),&\bar{X}_\mathrm{S}\leq\bar{X}_\mathrm{eut}\\
# T_\mathrm{m,FeS}(p)-\frac{\ln\left(\frac{1}{2}+\bar{X}_\mathrm{S}\right)}{\ln\left(\frac{1}{2}+\bar{X}_\mathrm{eut}\right)}(T_\mathrm{m,FeS}(p)-T_\mathrm{eut}(p)),&\text{else.}
# \end{cases}
# \end{equation}

# In[ ]:


def Tmalloy(p,X,mode):
    from math import log
    if X <= Xeut(p):
#       Fe-rich side of eutectic
        Tm=Tliq_Fe(p,mode)
        Tm=Tm-(Tm-Teut(p))*log(1-X)/log(1-Xeut(p))
    else:
#       Fe-poor side of eutectic
        Tm=Tliq_FeS(p)
        Tm=Tm-(Tm-Teut(p))*log(0.5+X)/log(0.5+Xeut(p))
    return Tm


# In practice, the fraction of the second component is usually given as a mass fraction $X_\mathrm{S}$ rather than as a mole fraction $\bar{X}_\mathrm{S}$. We convert from the former to the latter by
# \begin{equation}
# \bar{X}_\mathrm{S}=\left[1+\frac{u_\mathrm{S}}{u_\mathrm{Fe}}\left(\frac{1}{X_\mathrm{S}}-1\right)\right]^{-1},
# \end{equation}
# where $u_\mathrm{Fe}$ and $u_\mathrm{S}$ are the atomic masses of Fe and S, respectively. The atomic masses of elements here and elsewhere are taken from the IUPAC 2013 evaluation (Meija et al., 2016).

# In[ ]:


# mass/mole conversions for binary composites
def mass2mol(el,xel):
    if el in u.keys():
        return 1/(1.+u[el]/u['Fe']*(1./xel-1))
    else:
        raise NameError('Atomic mass for element '+el+' unavailable.')
def mol2mass(el,xel):
    if el in u.keys():
        return xel*u[el]/(xel*u[el]+(1-xel)*u['Fe'])
    else:
        raise NameError('Atomic mass for element '+el+' unavailable.')

# atomic masses (IUPAC 2013)
u={'O': 15.999e-3,'Na': 22.98976928e-3,'Mg': 24.305e-3,'S': 32.06e-3,   'K': 39.0983e-3,'Fe': 55.845e-3}


# ### <a id='main'></a>Main program (user interface)
# One can select the material (peridotite, basalt/eclogite, iron/iron alloy) and for peridotite also the system (terrestrial or martian peridotite, CMAS, or chondritic material, or else specify "other"). If the peridotite system "other" is selected, the mass fractions of MgO, FeO, Na<sub>2</sub>O, and K<sub>2</sub>O must be entered to perform the interpolation. If the alloys are selected, the mass fraction of S must be entered; if $X_\mathrm{S}=0$, only the data for pure Fe and no eutectic parameterization are shown.

# In[ ]:


import sys
from collections import defaultdict
import matplotlib.pyplot as plt
ox={}
p_crv=[]; Ts_crv=[]; Tl_crv=[]; Ti_crv=[]
ps_dat=defaultdict(list); pl_dat=defaultdict(list); Ts_dat=defaultdict(list); Tl_dat=defaultdict(list)
mat=int(input("Materials: peridotite (1), basalt/eclogite (2), iron/Fe-S alloy (3)\nSelect material: "))
plt.figure(figsize=(12,7))
if mat == 1:
#   peridotite
#   common liquidus phases: experimental data
#   forsterite
    pl_dat={'Davis & England (1964)': [0.55,0.55,1.25,1.25,1.8,2.5,3.05,3.95,3.95,4.65,4.65],
            'Navrotsky et al. (1989), Richet et al. (1993)': [0.,0.],
            'Ohtani & Kumazawa, 1981': [4.7,6.2,7.7,8.4,10.],
            'Presnall & Walter, 1993': [9.7,10.6,11.8,12.8,13.9,13.9,14.9,15.6,16.5,16.7]}
    Tl_dat={'Davis & England (1964)': [2178.,2203.,2223.,2248.,2253.,2303.,2303.,2353.,2378.,2378.,2403.],
            'Navrotsky et al. (1989), Richet et al. (1993)': [2163.,2174.],
            'Ohtani & Kumazawa, 1981': [2373.,2458.,2458.,2503.,2518.],
            'Presnall & Walter, 1993': [2483.,2493.,2513.,2533.,2583.,2543.,2553.,2583.,2553.,2588.]}
    mark_dat={'Davis & England (1964)': 'o','Navrotsky et al. (1989), Richet et al. (1993)': 'v',            'Ohtani & Kumazawa, 1981': '^','Presnall & Walter, 1993': '<'}
    for nm in pl_dat.keys():
        plt.plot(pl_dat[nm],Tl_dat[nm],linestyle='None',marker=mark_dat[nm],color='''0.2''')
    pl_dat={}; Tl_dat={}; mark_dat={}
#   pyrope
    pl_dat={'Boyd & England (1962)': [3.6,3.95,3.95,4.3,4.3,4.65,4.65],
            'Irifune & Ohtani (1986)': [3.5,3.5,5.,5.,5.5,5.5,6.25,6.25,7.,7.,7.5,7.5,8.,8.5,9.,9.5,10.],
            'Kudo & Ito (1996)': [15.,15.],
            'Ohtani et al. (1981)': [4.72,4.72,6.27,6.27,7.72,7.80,10.02],
            'Shen & Lazor (1995)': [8.4,11.,11.,14.,22.,20.,25.],
            'Zhang & Herzberg (1994)': [7.,8.,8.,9.,10.,10.,13.,16.,16.]}
    Tl_dat={'Boyd & England (1962)': [2048.,2048.,2073.,2073.,2098.,2098.,2123.],
            'Irifune & Ohtani (1986)': [2003.,2023.,2073.,2123.,2123.,2173.,2173.,2248.,2223.,2248.,2223.,2273.,2273.,2273.,2303.,2313.,2348.],
            'Kudo & Ito (1996)': [2523.,2573.],
            'Ohtani et al. (1981)': [2069.,2126.,2256.,2273.,2343.,2348.,2407.],
            'Shen & Lazor (1995)': [2300.,2450.,2480.,2650.,2830.,2800.,3100.],
            'Zhang & Herzberg (1994)': [2253.,2253.,2283.,2343.,2323.,2383.,2573.,2653.,2773.]}
    mark_dat={'Boyd & England (1962)': 'x','Irifune & Ohtani (1986)': 's',            'Kudo & Ito (1996)': '*','Ohtani et al. (1981)': '+',            'Shen & Lazor (1995)': 'D','Zhang & Herzberg (1994)': 'h'}
    for nm in pl_dat.keys():
        plt.plot(pl_dat[nm],Tl_dat[nm],linestyle='None',marker=mark_dat[nm],color='''0.5''')
    pl_dat={}; Tl_dat={}; mark_dat={}
#   periclase
    pl_dat={'Alfè (2005)': [0.4,17.,47.,135.6]}
    Tl_dat={'Alfè (2005)': [3070.,4590.,6047.,8144.]}
    mark_dat={'Alfè (2005)': '>'}
    for nm in pl_dat.keys():
        plt.plot(pl_dat[nm],Tl_dat[nm],linestyle='None',marker=mark_dat[nm],color='''0.8''')
    pl_dat={}; Tl_dat={}; mark_dat={}
#   common liquidus phases: parameterizations
    p_crv=np.linspace(0.,14.5,30)
    Tl_crv=[Tliqfo(p) for p in p_crv]
    plt.plot(p_crv,Tl_crv,color='''0.2''',linestyle="-.",label="Forsterite liquidus")
    p_crv=np.linspace(2.,26.,49)
    Tl_crv=[Tliqpy(p) for p in p_crv]
    plt.plot(p_crv,Tl_crv,color='''0.5''',linestyle="-.",label="Pyrope liquidus")
    p_crv=np.linspace(0.,140.,271)
    Tl_crv=[Tliqpc(p) for p in p_crv]
    plt.plot(p_crv,Tl_crv,color='''0.8''',linestyle="-.",label="Periclase liquidus")
#   lherzolite composition selection
    compsys=int(input("Compositions: terrestrial (1), martian (2), CMAS (3), chondritic (4), other (5):\n"                       "Select composition: "))
    if compsys == 1:
        title="Terrestrial peridotite"
#       experimental data and parameterization: batch liquidus
        pl_dat={'Fiquet et al. (2010)': [36.1,52.4,61.3,80.1,95.1,137.9],
                'Herzberg et al. (1990)': [14.,14.],
                'Ito & Takahashi (1987), upper mantle': [16.,18.,18.,20.,20.],
                'Ito & Takahashi (1987), lower mantle': [25.,25.],
                'Ito & Kennedy (1967)': [2.04],
                'Scarfe & Takahashi (1986)': [2.,2.,3.5,3.5,5.,5.,6.,7.,9.,9.,11.,13.],
                'Takahashi (1986)': [0.0001,1.5,1.5,3.,3.,5.,7.5,8.,8.,10.5,12.5,12.5,14.,14.],
                'Takahashi & Scarfe (1985)': [1.5,1.5,3.,5.,5.,7.5,11.,14.],
                'Takahashi et al. (1993)': [4.6,6.5],
                'Trønnes & Frost (2002)': [24.]}
        Tl_dat={'Fiquet et al. (2010)': [3714.,4101.,4300.,4453.,4801.,5401.],
                'Herzberg et al. (1990)': [2388.,2428.],
                'Ito & Takahashi (1987), upper mantle': [2273.,2381.,2473.,2482.,2682.],
                'Ito & Takahashi (1987), lower mantle': [2773.,3273.],
                'Ito & Kennedy (1967)': [2123.],
                'Scarfe & Takahashi (1986)': [1980.,2076.,2075.,2175.,2239.,2271.,2228.,2281.,2221.,2236.,2227.,2194.],
                'Takahashi (1986)': [1873.,2073.,2123.,2173.,2223.,2273.,2273.,2283.,2393.,2273.,2283.,2423.,2273.,2323.],
                'Takahashi & Scarfe (1985)': [2073.,2181.,2200.,2184.,2278.,2278.,2280.,2325.],
                'Takahashi et al. (1993)': [2173.,2273.],
                'Trønnes & Frost (2002)': [2613.]}
        mark_dat={'Fiquet et al. (2010)': '<','Herzberg et al. (1990)': 'o',                'Ito & Takahashi (1987), upper mantle': 'v','Ito & Takahashi (1987), lower mantle': 'v',                'Ito & Kennedy (1967)': '*','Scarfe & Takahashi (1986)': '^',                'Takahashi (1986)': '+','Takahashi & Scarfe (1985)': 'x',                'Takahashi et al. (1993)': 's','Trønnes & Frost (2002)': 'H'}
        for nm in pl_dat.keys():
            plt.plot(pl_dat[nm],Tl_dat[nm],linestyle='None',marker=mark_dat[nm],color='red')
        p_crv=np.linspace(0.,170.,341)
        Tl_crv=[Tliqb_per(p) for p in p_crv]
        plt.plot(p_crv,Tl_crv,"r--",label="Batch liquidus")
#       experimental data (lower mantle) and parameterization: solidus and fractional liquidus
        ps_dat={'Fiquet et al. (2010)': [36.2,51.8,61.3,80.2,95.1,106.6,122.0],
                'Hirose & Fei (2002)': [25.,25.,27.],
                'Ito & Takahashi (1987)': [25.],
                'Trønnes & Frost (2002)': [23.,24.5,23.,24.5,24.,24.5],
                'Zerr et al. (1997,1998)': [27.81,31.05,32.31,25.04,29.30,41.21,57.70,26.80,39.72,47.25,58.93],
                'Zhang & Herzberg (1994)': [22.5]}
        Ts_dat={'Fiquet et al. (2010)': [2817.,3124.,3196.,3690.,3930.,3908.,4100.],
                'Hirose & Fei (2002)': [2773.,2473.,2723.],
                'Ito & Takahashi (1987)': [2773.],
                'Trønnes & Frost (2002)': [2413.,2403.,2433.,2503.,2453.,2503.],
                'Zerr et al. (1997,1998)': [2924.,2994.,3128.,2272.,2495.,2832.,3255.,2880.,3139.,3404.,3456.],
                'Zhang & Herzberg (1994)': [2408.]}
        mark_dat={'Fiquet et al. (2010)': 'o','Hirose & Fei (2002)': 's',                  'Ito & Takahashi (1987)': 'v','Trønnes & Frost (2002)': '^',                  'Zerr et al. (1997,1998)': 'x','Zhang & Herzberg (1994)': '+'}
        Ts_crv=[Tsol_Earth(p) for p in p_crv]
        Tl_crv=[Tliq_per(p) for p in p_crv]
        pTrange=[0.,175.,1350.,8000.]
    elif compsys == 2:
        title="Martian peridotite"
#       experimental data and parameterization: solidus and fractional liquidus
        ps_dat={'Bertka & Holloway (1994)': [0.5,1.,1.,1.5,1.5,2.,2.,2.2,3.],
                'Collinet et al. (2015)': [1.,1.5,2.,2.2,0.5,0.5,1.,1.,1.5,1.5,2.,2.,2.2,2.2],
                'Matsukage et al. (2013)': [3.,4.5],
                'Schmerr et al. (2001)': [6.,6.,15.,15.,20.,23.,23.,25.,25.]}
        Ts_dat={'Bertka & Holloway (1994)': [1368.,1443.,1463.,1523.,1543.,1583.,1603.,1593.,1693.],
                'Collinet et al. (2015)': [1473.,1543.,1623.,1623.,1387.5,1387.5,1453.4,1453.4,1528.4,1528.4,1604.,1604.,1584.9,1584.9],
                'Matsukage et al. (2013)': [1639.,1743.],
                'Schmerr et al. (2001)': [1893.,1943.,2243.,2293.,2343.,2373.,2423.,2473.,2573.]}
        mark_dat={'Bertka & Holloway (1994)': 'o','Collinet et al. (2015)': 'x',                'Matsukage et al. (2013)': '^','Schmerr et al. (2001)': 'v'}
        p_crv=np.linspace(0.,25.,101)
        Tl_crv=[Tliq_per(p) for p in p_crv]
        Ts_crv=[Tsol_Mars(p) for p in p_crv]
        pTrange=[0.,25.,1350.,3000.]
    elif compsys == 3:
        title="CMAS composition"
#       experimental data and parameterization: solidus and fractional liquidus
        ps_dat={'Asahara et al. (1998)': [6.5],
                'Gudfinnsson & Presnall (1996)': [2.4,2.6,2.8,3.0,3.2,3.3,3.3,3.4],
                'Herzberg et al. (1990)': [9.2,14.,14.,15.3,15.3]}
        Ts_dat={'Asahara et al. (1998)': [2073.],
                'Gudfinnsson & Presnall (1996)': [1768.,1798.,1823.,1848.,1868.,1878.,1883.,1888.],
                'Herzberg et al. (1990)': [2243.,2328.,2393.,2333.,2458.]}
        mark_dat={'Asahara et al. (1998)': 'o','Gudfinnsson & Presnall (1996)': 'v',                 'Herzberg et al. (1990)': '^'}
        p_crv=np.linspace(0.,25.,101)
        Tl_crv=[Tliq_per(p) for p in p_crv]
        Ts_crv=[Tsol_CMAS(p) for p in p_crv]
        pTrange=[0.,25.,1400.,3000.]
    elif compsys == 4:
        title="Chondritic composition"
#       experimental data and parameterization: batch liquidus
        pl_dat={'Agee & Draper (2004)': [4.7,5.],
                'Andrault et al. (2011)': [20.7,27.9,35.5,42.5,45.1,50.2,59.5,65.0,68.1,69.9,84.9,110.6,141.1],
                'Berthet et al. (2009)': [1.,1.],
                'Ohtani (1987)': [12.,15.,17.5,20.],
                'Takahashi (1983)': [1.5]}
        Tl_dat={'Agee & Draper (2004)': [2123.,2123.],
                'Andrault et al. (2011)': [2525.,2724.,2947.,3117.,3021.,3153.,3308.,3499.,3499.,3643.,3946.,4397.,4777.],
                'Berthet et al. (2009)': [1823.,1873.],
                'Ohtani (1987)': [2283.,2303.,2363.,2423.],
                'Takahashi (1983)': [1923.]}
        mark_dat={'Agee & Draper (2004)': '+','Andrault et al. (2011)': 'o',
                'Berthet et al. (2009)': 's','Ohtani (1987)': '^','Takahashi (1983)': 'v'}
        for nm in pl_dat.keys():
            plt.plot(pl_dat[nm],Tl_dat[nm],linestyle='None',marker=mark_dat[nm],color='red')
        p_crv=np.linspace(0.,150.,301)
        Tl_crv=[Tliqb_chon(p) for p in p_crv]
        plt.plot(p_crv,Tl_crv,"r--",label="Batch liquidus")
#       experimental data and parameterization: solidus and fractional liquidus
        ps_dat={'Andrault et al. (2011)': [33.5,38.9,47.4,56.9,62.1,62.8,65.0,74.0,82.1,98.1,101.9,108.3,108.6,125.1,139.3,140.2],
                'Andrault et al. (2018)': [1.,11.6,4.7,9.8,17.4,1.4,7.,9.,16.,24.,13.5,2.2,28.,5.,5.,5.,5.,5.,20.],
                'Berthet et al. (2009)': [1.],
                'Ohtani (1987)': [12.,15.,17.5],
                'Takahashi (1983)': [0.8,1.5,2.25,0.6,0.6,1.5,1.5,2.5,2.5,3.]}
        Ts_dat={'Andrault et al. (2011)': [2608.,2656.,2697.,2850.,3060.,3102.,3002.,3248.,3342.,3555.,3705.,3800.,3647.,3997.,4147.,4202.],
                'Andrault et al. (2018)': [1519.,2030.,1823.,2006.,2155.,1600.,1839.,1917.,2107.,2268.,2062.,1616.,2351.,1865.,1840.,1840.,1815.,1815.,2185.],
                'Berthet et al. (2009)': [1473.],
                'Ohtani (1987)': [2073.,2173.,2198.],
                'Takahashi (1983)': [1473.,1473.,1573.,1423.,1523.,1473.,1573.,1573.,1673.,1673.]}
        mark_dat={'Andrault et al. (2011)': 'o','Andrault et al. (2018)': 'x',
                 'Berthet et al. (2009)': 's','Ohtani (1987)': '^','Takahashi (1983)': 'v'}
        Tl_crv=[Tliq_per(p) for p in p_crv]
        Ts_crv=[Tsol_chon(p) for p in p_crv]
        pTrange=[0.,150.,1400.,7000.]
    else:
        print("Oxide mass fractions (in %): ")
        sys.stdout.flush()
        for nm in ["MgO","FeO","Na2O","K2O"]:
#           convert % to fraction
            print("%s (Earth: %.2f%%; Mars: %.2f%%)" % (nm,100*ox_E[nm],100*ox_M[nm]))
            sys.stdout.flush()
            ox.update({nm: 1e-2*float(input("  Enter target "+nm+": "))})
#       parameterizations: interpolated solidus and fractional liquidus
        p_crv=np.linspace(0.,25.,101)
        Tl_crv=[Tliq_per(p) for p in p_crv]
        Ts_crv=[Tsol_intp(p,ox) for p in p_crv]
        pTrange=[0.,25.,1350.,3000.]
        title="Peridotitic composition with Mg#=%.3f" % Mg0
#   finish plotting of points and curves
    plt.plot(p_crv,Tl_crv,"r-",label="Fractional liquidus")
    if compsys <= 4:
        for nm in ps_dat.keys():
            plt.plot(ps_dat[nm],Ts_dat[nm],linestyle='None',marker=mark_dat[nm],color='blue')
    else:
#       parameterizations: terrestrial and martian or CMAS solidi as reference
        Ti_crv=[Tsol_Earth(p) for p in p_crv]
        plt.plot(p_crv,Ti_crv,"b--",label="terr. Solidus")
        if Mg0 < Mg0_E:
            Ti_crv=[Tsol_Mars(p) for p in p_crv]
            plt.plot(p_crv,Ti_crv,"b-.",label="mart. Solidus")
        else:
            Ti_crv=[Tsol_CMAS(p) for p in p_crv]
            plt.plot(p_crv,Ti_crv,"b-.",label="CMAS Solidus")
    plt.plot(p_crv,Ts_crv,"b-",label="Solidus")
elif mat == 2:
#   basalt/eclogite
    title="Basalt/eclogite"
#   common liquidus phases: experimental data
#   forsterite
    pl_dat={'Davis & England (1964)': [0.55,0.55,1.25,1.25,1.8,2.5,3.05,3.95,3.95,4.65,4.65],
            'Navrotsky et al. (1989), Richet et al. (1993)': [0.,0.],
            'Ohtani & Kumazawa, 1981': [4.7,6.2,7.7,8.4,10.],
            'Presnall & Walter, 1993': [9.7,10.6,11.8,12.8,13.9,13.9,14.9,15.6,16.5,16.7]}
    Tl_dat={'Davis & England (1964)': [2178.,2203.,2223.,2248.,2253.,2303.,2303.,2353.,2378.,2378.,2403.],
            'Navrotsky et al. (1989), Richet et al. (1993)': [2163.,2174.],
            'Ohtani & Kumazawa, 1981': [2373.,2458.,2458.,2503.,2518.],
            'Presnall & Walter, 1993': [2483.,2493.,2513.,2533.,2583.,2543.,2553.,2583.,2553.,2588.]}
    mark_dat={'Davis & England (1964)': 'o','Navrotsky et al. (1989), Richet et al. (1993)': 'v',            'Ohtani & Kumazawa, 1981': '^','Presnall & Walter, 1993': '<'}
    for nm in pl_dat.keys():
        plt.plot(pl_dat[nm],Tl_dat[nm],linestyle='None',marker=mark_dat[nm],color='''0.2''')
    pl_dat={}; Tl_dat={}; mark_dat={}
#   diopside
    pl_dat={'Boyd & England (1963)': [0.54,0.52,1.08,1.08,1.79,1.79,2.51,2.51,2.51,2.51,2.51,3.23,3.23,3.23,3.23,3.95,3.95,4.67,4.67],
            'Gasparik (1996)': [7.,8.,9.,10.,11.,12.,13.,14.,15.],
            'Richet & Bottinga (1984)': [0.],
            'Shen & Lazor (1995, di)': [10.,15.6],
            'Yoder (1952)': [0.,0.0285,0.0428,0.0429,0.0498,0.0498,0.0579,0.0581,0.0781,0.0781,0.0812,0.1003,0.1004,0.1014,0.1014,0.1014,0.1051,0.1107,0.1109,0.111,0.1116,0.1196,0.1371,0.1472,0.1475,0.1512,0.1514,0.1514,0.1797,0.1798,0.1978,0.1986,0.2039,0.2039,0.2041,0.2585,0.2587,0.259,0.3248,0.325,0.325,0.3722,0.421,0.4212,0.4212,0.4622,0.463,0.4987,0.4992,0.501]}
    Tl_dat={'Boyd & England (1963)': [1723.,1748.,1798.,1823.,1873.,1898.,1923.,1948.,1973.,1948.,1973.,1998.,2023.,1998.,2023.,2048.,2073.,2098.,2123.],
            'Gasparik (1996)': [2213.,2263.,2303.,2343.,2363.,2393.,2423.,2423.,2423.],
            'Richet & Bottinga (1984)': [1665.],
            'Shen & Lazor (1995, di)': [2390.,2500.],
            'Yoder (1952)': [1664.5,1669.,1670.2,1670.2,1671.,1671.,1672.4,1671.5,1674.3,1674.7,1674.9,1676.5,1675.9,1677.6,1679.4,1679.,1679.7,1676.9,1678.2,1676.9,1677.4,1680.7,1681.3,1682.3,1685.2,1683.9,1683.9,1684.3,1687.6,1687.6,1692.4,1691.9,1692.5,1691.3,1691.3,1697.5,1697.9,1699.9,1706.9,1708.5,1707.3,1712.2,1718.,1718.4,1718.8,1722.5,1724.9,1729.4,1728.6,1730.3]}
    mark_dat={'Boyd & England (1963)': 'x','Gasparik (1996)': '+',            'Richet & Bottinga (1984)': 's','Shen & Lazor (1995, di)': '*','Yoder (1952)': 'D'}
    for nm in pl_dat.keys():
        plt.plot(pl_dat[nm],Tl_dat[nm],linestyle='None',marker=mark_dat[nm],color='''0.4''')
    pl_dat={}; Tl_dat={}; mark_dat={}
#   pyrope
    pl_dat={'Boyd & England (1962)': [3.6,3.95,3.95,4.3,4.3,4.65,4.65],
            'Irifune & Ohtani (1986)': [3.5,3.5,5.,5.,5.5,5.5,6.25,6.25,7.,7.,7.5,7.5,8.,8.5,9.,9.5,10.],
            'Kudo & Ito (1996)': [15.,15.],
            'Ohtani et al. (1981)': [4.72,4.72,6.27,6.27,7.72,7.80,10.02],
            'Shen & Lazor (1995, py)': [8.4,11.,11.,14.,22.,20.,25.],
            'Zhang & Herzberg (1994)': [7.,8.,8.,9.,10.,10.,13.,16.,16.]}
    Tl_dat={'Boyd & England (1962)': [2048.,2048.,2073.,2073.,2098.,2098.,2123.],
            'Irifune & Ohtani (1986)': [2003.,2023.,2073.,2123.,2123.,2173.,2173.,2248.,2223.,2248.,2223.,2273.,2273.,2273.,2303.,2313.,2348.],
            'Kudo & Ito (1996)': [2523.,2573.],
            'Ohtani et al. (1981)': [2069.,2126.,2256.,2273.,2343.,2348.,2407.],
            'Shen & Lazor (1995, py)': [2300.,2450.,2480.,2650.,2830.,2800.,3100.],
            'Zhang & Herzberg (1994)': [2253.,2253.,2283.,2343.,2323.,2383.,2573.,2653.,2773.]}
    mark_dat={'Boyd & England (1962)': 'h','Irifune & Ohtani (1986)': '1',            'Kudo & Ito (1996)': '2','Ohtani et al. (1981)': '>',            'Shen & Lazor (1995, py)': '*','Zhang & Herzberg (1994)': '3'}
    for nm in pl_dat.keys():
        plt.plot(pl_dat[nm],Tl_dat[nm],linestyle='None',marker=mark_dat[nm],color='''0.6''')
    pl_dat={}; Tl_dat={}; mark_dat={}
#   Ca-perovskite
    pl_dat={'Gasparik et al. (1994)': [15.2],
            'Zerr et al. (1997)': [16.,17.7,20.,20.8,23.1,23.9,27.6,30.5,34.4,36.1,38.3,43.]}
    Tl_dat={'Gasparik et al. (1994)': [2783.],
            'Zerr et al. (1997)': [2860.,2920.,3005.,3035.,3190.,3185.,3450.,3590.,3905.,4030.,4005.,4120.]}
    mark_dat={'Gasparik et al. (1994)': '4','Zerr et al. (1997)': 'p'}
    for nm in pl_dat.keys():
        plt.plot(pl_dat[nm],Tl_dat[nm],linestyle='None',marker=mark_dat[nm],color='''0.8''')
    pl_dat={}; Tl_dat={}; mark_dat={}
#   common liquidus phases: parameterizations
    p_crv=np.linspace(0.,14.5,30)
    Tl_crv=[Tliqfo(p) for p in p_crv]
    plt.plot(p_crv,Tl_crv,color='''0.2''',linestyle="-.",label="Forsterite liquidus")
    p_crv=np.linspace(0.,16.,33)
    Tl_crv=[Tliqdi(p) for p in p_crv]
    plt.plot(p_crv,Tl_crv,color='''0.4''',linestyle="-.",label="Diopside liquidus")
    p_crv=np.linspace(2.,26.,49)
    Tl_crv=[Tliqpy(p) for p in p_crv]
    plt.plot(p_crv,Tl_crv,color='''0.6''',linestyle="-.",label="Pyrope liquidus")
    p_crv=np.linspace(14.5,45.,62)
    Tl_crv=[TliqCapv(p) for p in p_crv]
    plt.plot(p_crv,Tl_crv,color='''0.8''',linestyle="-.",label="Ca-perovskite liquidus")
#   experimental data and parameterizations: batch and fractional liquidi
    pl_dat={'Hirose & Fei (2002)': [22.],
            'Yasuda et al. (1994)': [3.,3.,5.,5.,6.,7.5,7.5,10.,10.,12.,14.,18.,20.,20.]}
    Tl_dat={'Hirose & Fei (2002)': [2523.],
            'Yasuda et al. (1994)': [1673.,1723.,1873.,1898.,1933.,2073.,2098.,2173.,2273.,2273.,2373.,2373.,2373.,2473.]}
    mark_dat={'Hirose & Fei (2002)': 'o','Yasuda et al. (1994)': 'x'}
    for nm in pl_dat.keys():
        plt.plot(pl_dat[nm],Tl_dat[nm],linestyle='None',marker=mark_dat[nm],color='red')
    p_crv=np.linspace(0.,150.,71)
    Tl_crv=[Tliqb_bas(p) for p in p_crv]
    plt.plot(p_crv,Tl_crv,"r--",label="Batch liquidus")
    Tl_crv=[Tliq_bas(p) for p in p_crv]
    plt.plot(p_crv,Tl_crv,"r-",label="Fractional liquidus")
#   experimental data and parameterization: solidus
    ps_dat={'Andrault et al. (2014)': [59.5,59.8,69.2,72.5,80.0,80.4,83.0,86.4,87.0,102.9,107.0,111.1,113.4,118.0,119.6,128.0,138.0,56.0,56.1,59.5,60.2,60.5,80.0,80.4,87.0,102.9,107.0,113.4,118.0,119.6,120.2,123.3,123.9,128.0,138.0],
            'Hirose et al. (1999)': [16.6,20.7,26.0,31.9,31.9,32.0,36.0,36.2,39.8,44.8,46.3,46.3,46.3,54.7,55.4,61.2,61.2,64.2,22.4,27.2],
            'Hirose & Fei (2002)': [22.,26.,27.,27.5],
            'Pradhan et al. (2015)': [48.2,58.8,79.4,89.5,102.9,114.2,128.5],
            'Yasuda et al. (1994)': [3.,3.,3.5,4.,5.,5.,7.5,7.5,10.,10.,12.,14.,14.,16.,18.,20.]}
    Ts_dat={'Andrault et al. (2014)': [3000.,2950.,3260.,3150.,3400.,3350.,3330.,3350.,3370.,3550.,3520.,3700.,3540.,3670.,3650.,3750.,3730.,3070.,2930.,3050.,2980.,3100.,3550.,3450.,3550.,3700.,3650.,3750.,3750.,3900.,3850.,3800.,3850.,3800.,3900.],
            'Hirose et al. (1999)': [2356.,2547.,2553.,2646.,2768.,2721.,2768.,2817.,2818.,2914.,2901.,3011.,3137.,3160.,3174.,3390.,3413.,3204.,2431.,2626.],
            'Hirose & Fei (2002)': [2423.,2673.,2703.,2703.],
            'Pradhan et al. (2015)': [2940.,3170.,3420.,3550.,3670.,3770.,3927.],
            'Yasuda et al. (1994)': [1523.,1573.,1698.,1573.,1773.,1823.,1923.,1973.,2023.,2073.,2123.,2223.,2273.,2273.,2373.,2373.]}
    mark_dat={'Andrault et al. (2014)': '3','Hirose et al. (1999)': '+',            'Hirose & Fei (2002)': 'o','Pradhan et al. (2015)': 's',            'Yasuda et al. (1994)': 'x'}
    for nm in ps_dat.keys():
        plt.plot(ps_dat[nm],Ts_dat[nm],linestyle='None',marker=mark_dat[nm],color='blue')
    Ts_crv=[Tsol_bas(p) for p in p_crv]
    plt.plot(p_crv,Ts_crv,"b-",label="Solidus")
    pTrange=[0.,150.,1200.,4000.]
elif mat == 3:
#   iron/iron alloys
    title="Fe"
#   experimental data: pure Fe
    pl_dat_Fe={'Strong et al. (1973), delta': [1.8,2.3,2.5,2.6,3.1,3.7,3.8,4.4,5.2,5.3],
               'Liu & Bassett (1975), delta': [2.86],
               'Swartzendruber (1982)': [0.],
               'Strong et al. (1973), fcc': [5.2,5.3,5.6,5.7,5.8],
               'Liu & Bassett (1975), fcc': [7.20,10.86,11.87,15.38,15.89,18.83,19.74],
               'Williams et al. (1987)': [12.,23.,38.,45.,55.,60.,64.,66.,68.,71.,82.,92.,10.,22.,68.],
               'Ringwood & Hibberson (1990)': [16.],
               'Boehler (1993), fcc': [6.,13.,15.,18.,19.,27.,34.,36.,39.,47.,48.,52.,64.,73.,81.],
               'Jephcoat & Besedin (1996)': [47.],
               'Shen et al. (1998), fcc': [12.0,23.2,23.2,29.2,30.3,39.2,42.2,46.3,46.3],
               'Ahrens et al. (2002)': [67.,70.,76.,79.],
               'Shen et al. (2004)': [27.,27.,42.,42.,50.,50.,57.,57.],
               'Boehler et al. (2008), fcc': [35.,50.,67.,42.,50.,54.,68.],
               'Buono & Walker (2011)': [6.],
               'Jackson et al. (2013)': [20.,28.,37.,40.,49.,50.,61.,82.],
               'Anzellini et al. (2013), fcc': [51.5,52.7,52.8,52.8,52.8,53.2,54.1,64.0,64.6,64.7,64.7,64.8,64.9,65.1,65.5,74.8,75.3,75.5,75.7,75.7,75.7,76.2,76.2,76.3,76.4,76.4,76.8,76.8,76.9,77.1,77.2,77.6,77.8,78.7,84.4,84.8,84.8,85.1,85.1,86.0,86.1,86.2,86.4,86.8,87.9],
               'Aquilanti et al. (2015), fcc': [74.9,75.5,79.9,80.4],
               'Zhang et al. (2016)': [19.0,19.1,55.9,57.1,59.8],
               'Boehler (1993), hcp': [93.,95.,106.,111.,111.,115.,117.,121.,121.,125.,126.,132.,134.,137.,139.,139.,143.,146.,146.,149.,149.,154.,154.,158.,158.,160.,167.,178.,185.,195.,195.,196.,197.],
               'Shen et al. (1998), hcp': [75.3,80.2],
               'Ma et al. (2004)': [60.,105.],
               'Nguyen & Holmes (2004)': [225.],
               'Boehler et al. (2008), hcp': [90.,116.,68.,85.,116.,130.],
               'Tateno et al. (2010)': [134.,135.5,210.,210.,330.,348.,356.,371.,376.],
               'Murphy et al. (2011)': [106.,121.,133.,151.],
               'Anzellini et al. (2013), hcp': [99.3,99.9,99.9,115.0,116.0,116.0,117.0,117.0,118.0,128.0,128.0,129.0,133.0,134.0,134.0,135.0,136.0,136.0,136.0,158.0,158.0,159.0,182.0,204.0],
               'Aquilanti et al. (2015), hcp': [93.0,93.5,116.4,116.8],
               'Boehler & Ross (2015)': [88.,93.,99.,101.]}
    Tl_dat_Fe={'Strong et al. (1973), delta': [1859.,1878.,1878.,1880.,1904.,1922.,1925.,1949.,1978.,1983.],
               'Liu & Bassett (1975), delta': [1912.3],
               'Swartzendruber (1982)': [1811.],
               'Strong et al. (1973), fcc': [1978.,1983.,1993.,2004.,2004.],
               'Liu & Bassett (1975), fcc': [2073.9,2161.1,2188.0,2247.5,2263.2,2326.1,2350.3],
               'Williams et al. (1987)': [2271.,2341.,2576.,2600.,2882.,2694.,3412.,3106.,3059.,3294.,3576.,3600.,2388.,2600.,3459.],
               'Ringwood & Hibberson (1990)': [2218.],
               'Boehler (1993), fcc': [2024.,2165.,2259.,2243.,2306.,2400.,2447.,2525.,2525.,2573.,2573.,2541.,2682.,2729.,2745.],
               'Jephcoat & Besedin (1996)': [2750.],
               'Shen et al. (1998), fcc': [2080.,2230.,2380.,2300.,2440.,2490.,2520.,2530.,2720.],
               'Ahrens et al. (2002)': [2698.,2776.,2776.,2729.],
               'Shen et al. (2004)': [2243.,2604.,2416.,2698.,2557.,2667.,2588.,2887.],
               'Boehler et al. (2008), fcc': [2381.,2543.,2736.,2597.,2713.,2674.,2844.],
               'Buono & Walker (2011)': [2078.],
               'Jackson et al. (2013)': [2210.,2280.,2500.,2400.,2560.,2650.,2840.,3025.],
               'Anzellini et al. (2013), fcc': [3031.,3137.,2994.,3026.,3205.,2938.,3139.,3225.,3277.,3084.,3202.,3314.,3044.,3342.,3312.,3534.,3337.,3323.,3352.,3400.,3529.,3532.,3566.,3460.,3507.,3504.,3560.,3400.,3368.,3319.,3473.,3425.,3425.,3376.,3635.,3771.,3664.,3549.,3763.,3458.,3598.,3535.,3618.,3597.,3650.],
               'Aquilanti et al. (2015), fcc': [2735.,2840.,2705.,2810.],
               'Zhang et al. (2016)': [2116.,2123.,2835.,2858.,2792.],
               'Boehler (1993), hcp': [2824.,2839.,2918.,2933.,3027.,3043.,3043.,2949.,3075.,3012.,3217.,3184.,3106.,3059.,3153.,3216.,3216.,3294.,3357.,3294.,3357.,3310.,3420.,3310.,3357.,3482.,3373.,3655.,3796.,3733.,3796.,3796.,3859.],
               'Shen et al. (1998), hcp': [2790.,2790.],
               'Ma et al. (2004)': [2750.,3510.],
               'Nguyen & Holmes (2004)': [5100.],
               'Boehler et al. (2008), hcp': [2952.,3045.,2844.,2975.,3176.,3168.],
               'Tateno et al. (2010)': [3244.,3083.,3853.,4109.,5135.,5327.,5519.,5423.,5679.],
               'Murphy et al. (2011)': [3520.,3880.,3930.,4160.],
               'Anzellini et al. (2013), hcp': [3521.,3615.,3842.,3846.,3892.,4069.,3791.,3841.,3911.,4064.,4119.,3961.,4097.,4024.,4269.,4095.,4114.,4175.,4087.,4269.,4463.,4612.,4669.,4923.],
               'Aquilanti et al. (2015), hcp': [2755.,2835.,3010.,3090.],
               'Boehler & Ross (2015)': [2792.,2871.,2933.,3043.]}
    mark_dat_Fe={'Strong et al. (1973), delta': 'v','Liu & Bassett (1975), delta': 'd','Swartzendruber (1982)': 'p',                 'Strong et al. (1973), fcc': 'v','Liu & Bassett (1975), fcc': 'd',                 'Williams et al. (1987)': 'x','Ringwood & Hibberson (1990)': 'p',                 'Boehler (1993), fcc': '2','Jephcoat & Besedin (1996)': 'H','Shen et al. (1998), fcc': '<',                 'Ahrens et al. (2002)': '^','Shen et al. (2004)': '<','Boehler et al. (2008), fcc': '>',                 'Buono & Walker (2011)': 'p','Jackson et al. (2013)': 's',                 'Anzellini et al. (2013), fcc': '.','Aquilanti et al. (2015), fcc': '+',                 'Zhang et al. (2016)': 'H',                 'Shen et al. (1998), hcp': '<','Boehler (1993), hcp': '2',                 'Ma et al. (2004)': '*','Nguyen & Holmes (2004)': 'p',                 'Boehler et al. (2008), hcp': '>','Tateno et al. (2010)': 'D',                 'Murphy et al. (2011)': 'o','Anzellini et al. (2013), hcp': '.',                 'Boehler & Ross (2015)': 'h','Aquilanti et al. (2015), hcp': '+'}
    for nm in pl_dat_Fe.keys():
        plt.plot(pl_dat_Fe[nm],Tl_dat_Fe[nm],linestyle='None',marker=mark_dat_Fe[nm],color='red')
#   parameterization
    p_crv=np.linspace(0.,400.,801)
    Tl_crv=[Tliq_Fe(p,'f') for p in p_crv]
    plt.plot(p_crv,Tl_crv,'r-',label="Fe, flat gradient")
    Tl_crv=[Tliq_Fe(p,'s') for p in p_crv]
    plt.plot(p_crv,Tl_crv,'r--',label="Fe, steep gradient")
#   decide whether to consider FeS and other alloys as well
    xS=0.
    wS=0.01*float(input("S content (wt.%): "))
    if wS > 0.:
        xS=mass2mol('S',wS)
        if xS > 0.5:
            print("WARNING: Alloy contains more S than Fe: xS=%.2f mol%%" % (xS*100))
            xS=0.5
            wS=mol2mass('S',xS)
            print("         Calculating curves for pure FeS: wS=%.2f wt.%%" % (wS*100))
            title+=", FeS, and eutectic"
        else:
            title+=", FeS, eutectic, and Fe + %.2f wt.%% (%.2f mol%%) S" % (wS*100,xS*100)
#       melting curve experimental data: pure FeS
        pl_dat_FeS={'STP': [0.],
                    'Anderson & Ahrens (1996)': [136.,330.,360.],
                    'Boehler (1992)': [6.4,10.8,11.7,22.1,31.1,43.0],
                    'Ono et al. (2008)': [43.,53.,64.,77.,95.,121.,154.,175.,223.,253.]}
        Tl_dat_FeS={'STP': [1467.],
                    'Anderson & Ahrens (1996)': [3240.,4210.,4310.],
                    'Boehler (1992)': [1702.,1787.,1802.,2067.,2182.,2348.],
                    'Ono et al. (2008)': [2406.,2483.,2594.,2710.,2839.,3004.,3191.,3325.,3697.,3953.]}
        mark_dat_FeS={'STP': 's','Anderson & Ahrens (1996)': 'x','Boehler (1992)': 'o',
                    'Ono et al. (2008)': '+'}
        for nm in pl_dat_FeS.keys():
            plt.plot(pl_dat_FeS[nm],Tl_dat_FeS[nm],linestyle='None',marker=mark_dat_FeS[nm],color='blue')
#       parameterization
        Tl_crv=[Tliq_FeS(p) for p in p_crv]
        plt.plot(p_crv,Tl_crv,'b-',label="FeS")
#       eutectic temperature experimental data
        pl_dat_eut={'Boehler (1996)': [5.],
                    'Brett & Bell (1969)': [3.],
                    'Buono & Walker (2011)': [6.],
                    'Campbell et al. (2007)': [29.6,31.4,36.6,52.1,54.,59.5,60.6,79.4,39.,59.7],
                    'Chen et al. (2008)': [10.,14.],
                    'Chudinovskikh & Boehler (2007)': [32.2,41.8,36.4],
                    'Fei et al. (1997, 2000)': [0.,7.,14.,18.,21.],
                    'Kamada et al. (2010)': [80.],
                    'Li et al. (2001)': [25.],
                    'Morard et al. (2007)': [6.1,8.85,10,10.6,13.21,13.35,13.6,15.3,16.5,16.97],
                    'Morard et al. (2008)': [29.33,31.87,32.17,43.,43.09,46.29,47.38,53.03,52.63,52.61,62.14,62.93,64.85],
                    'Mori et al. (2017)': [34.,46.,46.,60.,80.,130.,134.,200.,254.],
                    'Ryzhenko & Kennedy (1973)': [0.0001,0.0001,1.44,1.6,1.6,1.6,2.,2.,2.,2.,2.,2.,2.4,4.,4.,6.,6.,1.8,1.8,1.92,1.92,2.,2.,2.32,2.32,2.4,2.75,2.8,3.2,4.,4.,4.,4.,4.,6.,6.,6.,6.,3.,3.08,3.2,3.68,3.68,3.68,3.68,3.72,3.8,3.96,4.04,4.12,4.3,4.56,4.6,5.,5.2,5.36,5.48,5.68,5.76,5.8,5.96,6.06,2.,3.12,3.12,3.2,3.2,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.04,5.,5.12,5.2,5.8,5.96,6.,6.,6.,6.,6.,6.,6.],
                    'Stewart et al. (2007)': [23.,40.],
                    'Usselman (1975)': [3.,4.5,5.5]}
        Tl_dat_eut={'Boehler (1996)': [1318.],
                    'Brett & Bell (1969)': [1263.],
                    'Buono & Walker (2011)': [1263.],
                    'Campbell et al. (2007)': [1520.,1510.,1520.,1710.,1780.,1930.,1870.,2040.,1820.,2000.],
                    'Chen et al. (2008)': [1191.,1132.],
                    'Chudinovskikh & Boehler (2007)': [1500.,1596.,1547.],
                    'Fei et al. (1997, 2000)': [1261.,1197.,1133.,1133.,1348.],
                    'Kamada et al. (2010)': [2197.],
                    'Li et al. (2001)': [1424.],
                    'Morard et al. (2007)': [1140.,1160.,1200.,1100.,1180.,1250.,1200.,1100.,1270.,1120.],
                    'Morard et al. (2008)': [1560.,1422.,1479.,1660.,1732.,1730.,1795.,1720.,1813.,1864.,1911.,1966.,1997.],
                    'Mori et al. (2017)': [1630.,1770.,1900.,1910.,2050.,2910.,2960.,3320.,3550.],
                    'Ryzhenko & Kennedy (1973)': [1260.,1262.,1260.,1283.,1264.,1273.,1262.,1266.,1279.,1285.,1284.,1282.,1276.,1292.,1292.,1283.,1283.,1287.,1291.,1288.,1292.,1275.,1292.,1293.,1293.,1293.,1291.,1295.,1288.,1287.,1287.,1284.,1284.,1278.,1265.,1269.,1269.,1274.,1288.,1289.,1292.,1290.,1290.,1290.,1290.,1285.,1285.,1291.,1291.,1293.,1289.,1287.,1279.,1276.,1275.,1272.,1281.,1286.,1277.,1273.,1274.,1274.,1284.,1292.,1292.,1301.,1292.,1296.,1302.,1302.,1299.,1297.,1304.,1291.,1292.,1289.,1286.,1299.,1294.,1281.,1281.,1283.,1274.,1283.,1285.,1283.,1284.,1285.,1286.,1286.],
                    'Stewart et al. (2007)': [1312.,1508.],
                    'Usselman (1975)': [1265.,1267.,1278.]}
        mark_dat_eut={'Boehler (1996)': 'p','Brett & Bell (1969)': 'p','Buono & Walker (2011)': 'p',                    'Campbell et al. (2007)': 'd','Chen et al. (2008)': 'h',                    'Chudinovskikh & Boehler (2007)': 'o','Fei et al. (1997, 2000)': '+',
                    'Kamada et al. (2010)': 'p','Li et al. (2001)': 'p',\
                    'Morard et al. (2007)': 'x','Morard et al. (2008)': 'x',\
                    'Mori et al. (2017)': 's','Ryzhenko & Kennedy (1973)': '2',
                    'Stewart et al. (2007)': 'D','Usselman (1975)': 'v'}
        for nm in pl_dat_eut.keys():
            plt.plot(pl_dat_eut[nm],Tl_dat_eut[nm],linestyle='None',marker=mark_dat_eut[nm],color='cyan')
#       eutectic temperature parameterization
        Tl_crv=[Teut(p) for p in p_crv]
        plt.plot(p_crv,Tl_crv,'c-',label="Eutectic")
        if xS < 0.5:
            Tl_crv=[Tmalloy(p,xS,'f') for p in p_crv]
            plt.plot(p_crv,Tl_crv,'k-',label="Alloy, flat Fe liq.")
            Tl_crv=[Tmalloy(p,xS,'s') for p in p_crv]
            plt.plot(p_crv,Tl_crv,'k--',label="Alloy, steep Fe liq.")
        pTrange=[0.,400.,1000.,6000.]
    else:
        pTrange=[0.,400.,1500.,6000.]
else:
    raise IOError("Unknown material.")
# finish p-T plot
plt.axis(pTrange)
plt.xlabel('$p$ (GPa)')
plt.ylabel('$T_\mathrm{s,l}$ (K)')
plt.legend()
plt.title(title)
plt.show()

# also show information about the eutectic composition for alloys
if mat == 3 and xS > 0 and xS < 0.5:
    plt.figure(figsize=(12,7))
#   eutectic composition experimental data
    Xe_crv=[]; Xe_dat_eut=defaultdict(list)
    del pl_dat_eut['Boehler (1996)']
    pl_dat_eut['Campbell et al. (2007)']=[29.6,31.4,36.6]
    pl_dat_eut['Fei et al. (1997, 2000)']=[0.,7.,14.,21.]
    pl_dat_eut['Morard et al. (2008)']=[64.85]
    pl_dat_eut['Ryzhenko & Kennedy (1973)']=[4.,6.]
    Xe_dat_eut={'Brett & Bell (1969)': [0.271],
                'Buono & Walker (2011)': [0.302],
                'Campbell et al. (2007)': [0.25,0.25,0.25],
                'Chen et al. (2008)': [0.30337,0.2766],
                'Chudinovskikh & Boehler (2007)': [0.20942,0.189,0.20072],
                'Fei et al. (1997, 2000)': [0.43902,0.31257,0.27931,0.154],
                'Kamada et al. (2010)': [0.20072],
                'Li et al. (2001)': [0.23],
                'Morard et al. (2007)': [0.35,0.35,0.35,0.38,0.33,0.33,0.33,0.3,0.3,0.3],
                'Morard et al. (2008)': [0.187],
                'Mori et al. (2017)': [0.19926,0.21662,0.22235,0.19487,0.18161,0.16818,0.14849,0.12376,0.09526],
                'Ryzhenko & Kennedy (1973)': [0.36734,0.32945],
                'Stewart et al. (2007)': [0.25191,0.19487],
                'Usselman (1975)': [0.39303,0.37229,0.35988]}
    for nm in pl_dat_eut.keys():
        Xe_dat_eut[nm]=[100*x for x in Xe_dat_eut[nm]]
        plt.plot(pl_dat_eut[nm],Xe_dat_eut[nm],linestyle='None',marker=mark_dat_eut[nm],color='cyan')
#   eutectic composition parameterization
    Xe_crv=[100*Xeut(p) for p in p_crv]
    plt.plot(p_crv,Xe_crv,'c-')
    pXrange=[0.,400.,5.,40.]
    plt.axis(pXrange)
    plt.ylabel('$X_\mathrm{e,S}$ (mol%)')
    plt.title('Eutectic composition for Fe-FeS')
    plt.show()


# #### <a id='hist'></a>Version history
# - v.1.3 (14/10/2018):
#     - Removed jump between transition zone and lower mantle assemblages in terrestrial peridotite by refitting data with a continuity constraint at 23.5 GPa. The data point from Ito & Katsura (1992) and some data points from Hirose & Fei (2002), Trønnes & Frost (2002) and Zerr et al. (1998) at $p$ below 22.5 GPa that had been used in previous versions were excluded now.
#     - As a consequence of the new terrestrial lower-mantle solidus, the shift applied to for the CMAS assemblage has been increased to about 139 K, ensuring continuity here as well.
# - v.1.2 (28/06/2018):
#     - Port to Python 3
#     - Added solidus and liquidus for chondritic composition
#     - Limited alloying element in iron alloys to at most 50 mol% (i.e., alloys are restricted to the Fe-rich side of the Fe-S binary system)
#     - Added reference to Iwamori et al. (1995) for liquidus discussion
# - v.1.1 (29/11/2017):
#     - Refit of the eclogite system, with experimental data replacing most of the Litasov & Ohtani curves at $p\gtrsim 3$ GPa and extending the solidus to the terrestrial CMB.
#     - Minor update of the lower-mantle terrestrial solidus (Hirose & Fei, 2002)
#     - Some additions and small corrections to the text
# - v.1.0 (22/11/2017): Original version

# ### <a id='ref'></a>References
# - Agee, C. B.; Draper, D. S. (2004): Experimental constraints on the origin of Martian meteorites and the composition of the Martian mantle. Earth Planet. Sci. Lett. 224(3-4), 415-429, [doi:10.1016/j.epsl.2004.05.022](http://dx.doi.org/10.1016/j.epsl.2004.05.022)
# - Ahrens, T. J.; Holland, K. G.; Chen, G. Q. (2002): Phase diagram of iron, revised-core temperatures. Geophys. Res. Lett. 29(7), 1150, [doi:10.1029/2001GL014350](http://dx.doi.org/10.1029/2001GL014350)
# - Alfè, D. (2005): Melting curve of MgO from first-principles simulations. Phys. Rev. Lett. 94, 235701, [doi:10.1103/PhysRevLett.94.235701](http://dx.doi.org/10.1103/PhysRevLett.94.235701)
# - Anderson, O. L. (1998): The Grüneisen parameter for iron at outer core conditions and the resulting conductive heat and power in the core. Phys. Earth Planet. Inter. 109(3-4), 179-197, [doi:10.1016/S0031-9201(98)00123-X](http://dx.doi.org/10.1016/S0031-9201(98)00123-X)
# - Anderson, W. W.; Ahrens, T. J. (1996): Shock temperature and melting in iron sulfides at core pressures. J. Geophys. Res. 101(B3), 5627-5642, [doi:10.1029/95JB01972](http://dx.doi.org/10.1029/95JB01972)
# - Andrault, D.; Bolfan-Casanova, N.; Lo Nigro, G.; Bouhifd, M. A.; Garbarino, G.; Mezouar, M. (2011): Solidus and liquidus profiles of chondritic mantle: Implication for melting of the Earth across its history. Earth Planet. Sci. Lett. 304(1-2), 251-259, [doi:10.1016/j.epsl.2011.02.006](http://dx.doi.org/10.1016/j.epsl.2011.02.006)
# - Andrault, D.; Pesce, G.; Bouhifd, M. A.; Bolfan-Casanova, N.; Hénot, J.-M.; Mezouar, M. (2014): Melting of subducted basalt at the core-mantle boundary. Science 344(6186), 892-895, [doi:10.1126/science.1250466](http://dx.doi.org/10.1126/science.1250466)
# - Andrault, D.; Pesce, G.; Manthilake, G.; Monteux, J.; Bolfan-Casanova, N.; Chantel, J.; Novella, D.; Guignot, N.; King, A.; Itié, J.-P.; Hennet, L. (2018): Deep and persistent melt layer in the Archaean mantle. Nature Geosci. 11(2), 139-143, [doi:10.1038/s41561-017-0053-9](http://dx.doi.org/10.1038/s41561-017-0053-9)
# - Anzellini, S.; Dewaele, A.; Mezouar, M.; Loubeyre, P.; Morard, G. (2013): Melting of iron at Earth's inner core boundary based on fast X-ray diffraction. Science 340(6131), 464-466, [doi:10.1126/science.1233514](http://dx.doi.org/10.1126/science.1233514)
# - Aquilanti, G.; Trapananti, A.; Karandikar, A.; Kantor, I.; Marini, C.; Mathon, O.; Pascarelli, S.; Boehler, R. (2015): Melting of iron determined by X-ray absorption spectroscopy to 100 GPa. Proc. Nat. Acad. Sci. 112(39), 12042-12045, [doi:10.1073/pnas.1502363112](http://dx.doi.org/10.1073/pnas.1502363112)
# - Asahara, Y.; Ohtani, E.; Suzuki, A. (1998): Melting relations of hydrous and dry mantle compositions and the genesis of komatiites. Geophys. Res. Lett. 25(12), 2201-2204, [doi:10.1029/98GL01527](http://dx.doi.org/10.1029/98GL01527)
# - Asahara, Y.; Ohtani, E. (2001): Melting relations of the hydrous primitive mantle in the CMAS-H<sub>2</sub>O system at high pressures and temperatures, and implications for generation of komatiites. Phys. Earth Planet. Inter. 125(1-4), 31-44, [doi:10.1016/S0031-9201(01)00208-4](http://dx.doi.org/10.1016/S0031-9201(01)00208-4)
# - Berthet, S.; Malavergne, V.; Righter, K. (2009): Melting of the Indarch meteorite (EH4 chondrite) at 1 GPa and variable oxygen fugacity: Implications for early planetary differentiation processes. Geochim. Cosmochim. Acta 73(20), 6402-6420, [doi:10.1016/j.gca.2009.07.030](http://dx.doi.org/10.1016/j.gca.2009.07.030)
# - Bertka, C. M.; Holloway, J. R. (1994): Anhydrous partial melting of an iron-rich mantle I: subsolidus phase assemblages and partial melting phase relations at 10 to 30 kbar. Contrib. Mineral. Petrol. 115(3), 313-322, [doi:10.1007/BF00310770](http://dx.doi.org/10.1007/BF00310770)
# - Boehler, R. (1992): Melting of the Fe-FeO and the Fe-FeS systems at high pressure: Constraints on core temperatures. Earth Planet. Sci. Lett. 111(2-4), 217-227, [doi:10.1016/0012-821X(92)90180-4](http://dx.doi.org/10.1016/0012-821X(92)90180-4)
# - Boehler, R. (1993): Temperatures in the Earth's core from melting-point measurements of iron at high static pressures. Nature 363, 534-536, [doi:10.1038/363534a0](http://dx.doi.org/10.1038/363534a0)
# - Boehler, R. (1996): Fe-FeS eutectic temperatures to 620 kbar. Phys. Earth Planet. Inter. 96(2-3), 181-186, [doi:10.1016/0031-9201(96)03150-0](http://dx.doi.org/10.1016/0031-9201(96)03150-0)
# - Boehler, R.; Santamaría-Pérez, D.; Errandonea, D.; Mezouar, M. (2008): Melting, density, and anisotropy of iron at core conditions: new x-ray measurements to 150 GPa. J. Phys. Conf. Ser. 121(2), 022018, [doi:10.1088/1742-6596/121/2/022018](http://dx.doi.org/10.1088/1742-6596/121/2/022018)
# - Boehler, R.; Ross, M. (2015): Properties of rocks and minerals, high-pressure melting. In: Mineral Physics, ed. by Price, G. D. and Stixrude, L., vol. 2 in Treatise on Geophysics, Elsevier, pp. 573-582, [doi:10.1016/B978-0-444-53802-4.00046-4](http://dx.doi.org/10.1016/B978-0-444-53802-4.00046-4)
# - Brett, R.; Bell, P. M. (1969): Melting relations in the Fe-rich portion of the system Fe-FeS at 30 kb pressure. Earth Planet. Sci. Lett. 6(6), 479-482, [doi:10.1016/0012-821X(69)90119-8](http://dx.doi.org/10.1016/0012-821X(69)90119-8)
# - Buono, A.; Walker, D. (2011): The Fe-rich liquidus in the Fe-FeS system from 1 bar to 10 GPa. Geochim. Cosmochim. Acta 75(8), 2072-2087, [doi:10.1016/j.gca.2011.01.030](http://dx.doi.org/10.1016/j.gca.2011.01.030)
# - Boyd, Jr., F. R.; England, J. L. (1962): [Mantle minerals](https://archive.org/stream/yearbookcarne61196162carn#page/106/mode/2up). Year Book 1961-1962, vol.61, 107-112, Carnegie Institution of Washington
# - Boyd, F. R.; England, J. L. (1963): Effect of pressure on the melting of diopside, CaMgSi<sub>2</sub>O<sub>6</sub>, and albite, NaAlSi<sub>3</sub>O<sub>8</sub>, in the range up to 50 kilobars. J. Geophys. Res. 68(1), 311-323, [doi:10.1029/JZ068i001p00311](http://dx.doi.org/10.1029/JZ068i001p00311)
# - Campbell, A. J.; Seagle, C. T.; Heinz, D. L.; Shen, G.; Prakapenka, V. B. (2007): Partial melting in the iron-sulfur system at high pressure: A synchrotron X-ray diffraction study. Phys. Earth Planet. Inter. 162(1-2), 119-128, [doi:10.1016/j.pepi.2007.04.001](http://dx.doi.org/10.1016/j.pepi.2007.04.001)
# - Cartier, C.; Hammouda, T.; Doucelance, R.; Boyet, M.; Devidal, J.-L.; Moine, B. (2014): Experimental study of trace element partitioning between enstatite and melt in enstatite-chondrites at low oxygen fugacities and 5 GPa. Geochim. Cosmochim. Acta 130, 167-187, [doi:10.1016/j.epsl.2014.05.008](http://dx.doi.org/10.1016/j.epsl.2014.05.008)
# - Chen, B.; Li, J.; Hauck II, S. A. (2008): Non-ideal liquidus curve in the Fe-S system and Mercury's snowing core. Geophys. Res. Lett. 35, L07201, [doi:10.1029/2008GL033311](http://dx.doi.org/10.1029/2008GL033311)
# - Chudinovskikh, L.; Boehler, R. (2007): Eutectic melting in the system Fe-S to 44 GPa. Earth Planet. Sci. Lett. 257(1-2), 97-103, [doi:10.1016/j.epsl.2007.02.024](http://dx.doi.org/10.1016/j.epsl.2007.02.024)
# - Collinet, M.; Médard, E.; Charlier, B.; Vander Auwera, J.; Grove, T. L. (2015): Melting of the primitive martian mantle at 0.5-2.2 GPa and the origin of basalts and alkaline rocks on Mars. Earth Planet. Sci. Lett. 427, 83-94, [doi:10.1016/j.epsl.2015.06.056](http://dx.doi.org/10.1016/j.epsl.2015.06.056)
# - Davis, B. T. C.; England, J. L. (1964): The melting of forsterite up to 50 kilobars. J. Geophys. Res. 69(6), 1113-1116, [doi:10.1029/JZ069i006p01113](http://dx.doi.org/10.1029/JZ069i006p01113)
# - Fei, Y.; Bertka, C. M.; Finger, L. W. (1997): High-pressure iron-sulfur compound, Fe<sub>3</sub>S<sub>2</sub>, and melting relations in the Fe-FeS system. Science 275, 1621-1623, [doi:10.1126/science.275.5306.1621](http://dx.doi.org/10.1126/science.275.5306.1621)
# - Fei, Y.; Li, J.; Bertka, C. M.; Rewitt, C. T. (2000): [Structure type and bulk modulus of Fe<sub>3</sub>S, a new iron-sulfur compound](http://www.minsocam.org/ProcessIP.lasso?year=2000&filename=Fei_p1830-1833_00.pdf). Amer. Mineral. 85, 1830-1833
# - Fiquet, G.; Auzende, A. L.; Siebert, J.; Corgne, A.; Bureau, H.; Ozawa, H.; Garbarino, G. (2010): Melting of peridotite to 140 gigapascals. Science 329(5998), 1516-1518, [doi:10.1126/science.1192448](http://dx.doi.org/10.1126/science.1192448)
# - Gasparik, T. (1996): Melting experiments on the enstatite-diopside join at 70-224 kbar, including the melting of diopside. Contrib. Mineral. Petrol. 124(2), 139-153, [doi:10.1007/s004100050181](http://dx.doi.org/10.1007/s004100050181)
# - Gasparik, T.; Wolf, K.; Smith, C. M. (1994): [Experimental determination of phase relations in the CaSiO<sub>3</sub> system from 8 to 15 GPa](http://www.minsocam.org/ammin/AM79/AM79_1219.pdf). Amer. Mineral. 79(11-12), 1219-1222
# - Gudfinnsson, G. H.; Presnall, D. C. (1996): Melting relations of model lherzolite in the system CaO-MgO-Al<sub>2</sub>O<sub>3</sub>-SiO<sub>2</sub> at 2.4-3.4 GPa and the generation of komatiites. J. Geophys. Res. 101(B12), 27701-27710, [doi:10.1029/96JB02462](http://dx.doi.org/10.1029/96JB02462)
# - Gudfinnsson, G. H.; Presnall, D. C. (2000): Melting behaviour of model lherzolite in the system CaO-MgO-Al<sub>2</sub>O<sub>3</sub>-SiO<sub>2</sub>-FeO at 0.7-2.8 GPa. J. Petrol. 41(8), 1241-1269, [doi:10.1093/petrology/41.8.1241](http://dx.doi.org/10.1093/petrology/41.8.1241)
# - Herzberg, C.; Gasparik, T.; Sawamoto, H. (1990): Origin of mantle peridotite: Constraints from melting experiments to 16.5 GPa. J. Geophys. Res. 95(B10), 15779-15803, [doi:10.1029/JB095iB10p15779](http://dx.doi.org/10.1029/JB095iB10p15779)
# - Hirose, K.; Fei, Y. (2002): Subsolidus and melting phase relations of basaltic composition in the uppermost lower mantle. Geochim. Cosmochim. Acta 66(12), 2099-2108, [doi:10.1016/S0016-7037(02)00847-5](http://dx.doi.org/10.1016/S0016-7037(02)00847-5)
# - Hirose, K.; Fei, Y.; Ma, Y.; Mao, H.-K. (1999): The fate of subducted basaltic crust in the Earth's lower mantle. Nature 397, 53-56, [doi:10.1038/16225](http://dx.doi.org/10.1038/16225)
# - Hirschmann, M. M. (2000): Mantle solidus: Experimental constraints and the effects of peridotite composition. Geochem. Geophys. Geosyst. 1, 1042, [doi:10.1029/2000GC000070](http://dx.doi.org/10.1029/2000GC000070)
# - Hirschmann, M. M.; Tenner, T.; Aubaud, C.; Withers, A. C. (2009): Dehydration melting of nominally anhydrous mantle: The primacy of partitioning. Phys. Earth Planet. Inter. 176(1-2), 54-68, [doi:10.1016/j.pepi.2009.04.001](http://dx.doi.org/10.1016/j.pepi.2009.04.001)
# - Irifune, T.; Ohtani, E. (1986): Melting of pyrope Mg<sub>3</sub>Al<sub>2</sub>Si<sub>3</sub>O<sub>12</sub> up to 10 GPa: possibility of a pressure-induced structural change in pyrope melt. J. Geophys. Res. 91(B9), 9357-9366, [doi:10.1029/JB091iB09p09357](http://dx.doi.org/10.1029/JB091iB09p09357)
# - Ito, E.; Katsura, T. (1992): Melting of ferromagnesian silicates under the lower mantle conditions. In: High-Pressure Research: Application to Earth and Planetary Sciences, ed. by Syono, Y. and Manghnani, M. H. Terra Scientific Publishing Company (TERRAPUB)/American Geophysical Union, pp. 315-322, [doi:10.1029/GM067p0315](http://dx.doi.org/10.1029/GM067p0315)
# - Ito, E.; Takahashi, E. (1987): Melting of peridotite at uppermost lower-mantle conditions. Nature 328(6130), 514-517, [doi:10.1038/328514a0](http://dx.doi.org/10.1038/328514a0)
# - Ito, K.; Kennedy, G. C. (1967): Melting and phase relations in a natural peridotite to 40 kilobars. Am. J. Sci. 265, 519-538, [doi:10.2475/ajs.265.6.519](http://dx.doi.org/10.2475/ajs.265.6.519)
# - Iwamori, H.; McKenzie, D.; Takahashi, E. (1995): Melt generation by isentropic mantle upwelling. Earth Planet. Sci. Lett. 134, 253-266, [doi:10.1016/0012-821X(95)00122-S](http://dx.doi.org/10.1016/0012-821X(95)00122-S)
# - Jackson, J. M.; Sturhahn, W.; Lerche, M. J.; Zhao, J.; Toellner, T. S.; Alp, E. E.; Sinogeikin, S. V.; Bass, J. D.; Murphy, C. A.; Wicks, J. K. (2013): Melting of compressed iron by monitoring atomic dynamics. Earth Planet. Sci. Lett. 362, 143-150, [doi:10.1016/j.epsl.2012.11.048](http://dx.doi.org/10.1016/j.epsl.2012.11.048)
# - Jephcoat, A. P.; Besedin, S. P. (1996): Temperature measurements and melting determination in the laser-heated diamond-anvil cell. Phil. Trans. R. Soc. 354(1711), 1333-1360, [doi:10.1098/rsta.1996.0051](http://dx.doi.org/10.1098/rsta.1996.0051)
# - Kamada, S.; Terasaki, H.; Ohtani, E.; Sakai, T.; Kikegawa, T.; Ohishi, Y.; Hirao, N.; Sata, N.; Kondo, T. (2010): Phase relationships of the Fe-FeS system in conditions up to the Earth's outer core. Earth Planet. Sci. Lett. 294(1-2), 94-100, [doi:10.1016/j.epsl.2010.03.011](http://dx.doi.org/10.1016/j.epsl.2010.03.011)
# - Katz, R. F.; Spiegelman, M.; Langmuir, C. (2003): A new parameterization of hydrous mantle melting. Geochem. Geophys. Geosyst. 4(9), 1073, [doi:10.1029/2002GC000433](http://dx.doi.org/10.1029/2002GC000433)
# - Klemme, S.; O'Neill, H. St. C. (2000): The near-solidus transition from garnet lherzolite to spinel lherzolite. Contrib. Mineral. Petrol. 138, 237-248, [doi:10.1007/s004100050560](http://dx.doi.org/10.1007/s004100050560)
# - Kudo, R.; Ito, E. (1996): Melting relations in the system Mg<sub>4</sub>Si<sub>4</sub>O<sub>12</sub>-Mg<sub>3</sub>Al<sub>2</sub>Si<sub>3</sub>O<sub>12</sub> (Py) at high pressures. Phys. Earth Planet. Inter. 96(2-3), 159-169, [doi:10.1016/0031-9201(96)03148-2](http://dx.doi.org/10.1016/0031-9201(96)03148-2)
# - Li, J.; Fei, Y.; Mao, H. K.; Hirose, K.; Shieh, S. R. (2001): Sulfur in the Earth's inner core. Earth Planet. Sci. Lett. 193(3-4), 509-514, [doi:10.1016/S0012-821X(01)00521-0](http://dx.doi.org/10.1016/S0012-821X(01)00521-0)
# - Litasov, K.; Ohtani, E. (2002): Phase relations and melt compositions in CMAS-pyrolite-H<sub>2</sub>O system up to 25 GPa. Phys. Earth Planet. Inter. 134(1-2), 105-127, [doi:10.1016/S0031-9201(02)00152-8](http://dx.doi.org/10.1016/S0031-9201(02)00152-8)
# - Litasov, K. D.; Ohtani, E. (2007): Effect of water on the phase relations in Earth's mantle and deep water cycle. In: Advances in High-Pressure Mineralogy, ed. by Ohtani, E., Geological Society of America Special Paper 421, Boulder, Colorado, pp. 115-146, [doi:10.1130/2007.2421(08)](http://dx.doi.org/10.1130/2007.2421(08))
# - Liu, L.-G.; Bassett, W. A. (1975): The melting of iron up to 200 kbar. J. Geophys. Res. 80(26), 3777-3782, [doi:10.1029/JB080i026p03777](http://dx.doi.org/10.1029/JB080i026p03777)
# - Liu, X.; O'Neill, H. St. C. (2004): Partial melting of spinel lherzolite in the system CaO-MgO-Al<sub>2</sub>O<sub>3</sub>-SiO<sub>2</sub>±K<sub>2</sub>O at 1.1 GPa. J. Petrol. 45(7), 1339-1368 [doi:10.1093/petrology/egh021](http://dx.doi.org/10.1093/petrology/egh021)
# - Ma, Y.; Somayazulu, M.; Shen, G.; Mao, H.-k.; Shu, J.; Hemley, R. J. (2004): In situ X-ray diffraction studies of iron to Earth-core conditions. Phys. Earth Planet. Inter. 143-144, 455-467, [doi:10.1016/j.pepi.2003.06.005](http://dx.doi.org/10.1016/j.pepi.2003.06.005)
# - Matsukage, K. N.; Nagayo, Y.; Whitaker, M. L.; Takahashi, E.; Kawasaki, T. (2013): Melting of the Martian mantle from 1.0 to 4.5 GPa. J. Miner. Petr. Sci. 108(4), 201-214, [doi:10.2465/jmps.120820](http://dx.doi.org/10.2465/jmps.120820)
# - Meija, J.; Coplen, T. B.; Berglund, M.; Brand, W. A.; De Bièvre, P.; Gröning, M.; Holden, N. E.; Irrgeher, J.; Loss, R. D.; Walczyk, T.; Prohaska, T. (2016): Atomic weights of the elements 2013 (IUPAC Technical Report). Pure Appl. Chem. 88(3), 265-291, [doi:10.1515/pac-2015-0305](http://dx.doi.org/10.1515/pac-2015-0305)
# - Milholland, C. S.; Presnall, D. C. (1998): Liquidus phase relations in the CaO-MgO-Al<sub>2</sub>O<sub>3</sub>-SiO<sub>2</sub> system at 3.0 GPa: the aluminous pyroxene thermal divide and high-pressure fractionation of picritic and komatiitic magmas. J. Petrol. 39(1), 3-27, [doi:10.1093/petroj/39.1.3](http://dx.doi.org/10.1093/petroj/39.1.3)
# - Morard, G.; Sanloup, C.; Fiquet, G.; Mezouar, M.; Rey, N.; Poloni, R.; Beck, P. (2007): Structure of eutectic Fe-FeS melts to pressures up to 17 GPa: Implications for planetary cores. Earth Planet. Sci. Lett. 263(1-2), 128-139, [doi:10.1016/j.epsl.2007.09.009](http://dx.doi.org/10.1016/j.epsl.2007.09.009)
# - Morard, G.; Andrault, D.; Guignot, N.; Sanloup, C.; Mezouar, M.; Petitgirard, S.; Fiquet, G. (2008): *In situ* determination of Fe-Fe<sub>3</sub>S phase diagram and liquid structural properties up to 65 GPa. Earth Planet. Sci. Lett. 272(3-4), 620-626, [doi:10.1016/j.epsl.2008.05.028](http://dx.doi.org/10.1016/j.epsl.2008.05.028)
# - Mori, Y.; Ozawa, H.; Hirose, K.; Sinmyo, R.; Tateno, S.; Morard, G.; Ohishi, Y. (2017): Melting experiments on Fe-Fe<sub>3</sub>S system to 254 GPa. Earth Planet. Sci. Lett. 464, 135-141, [doi:10.1016/j.epsl.2017.02.021](http://dx.doi.org/10.1016/j.epsl.2017.02.021)
# - Murphy, C. A.; Jackson, J. M.; Sturhahn, W.; Chen, B. (2011): Melting and thermal pressure of hcp-Fe from the phonon density of states. Phys. Earth Planet. Inter. 188(1-2), 114-120, [doi:10.1016/j.pepi.2011.07.001](http://dx.doi.org/10.1016/j.pepi.2011.07.001)
# - Navrotsky, A.; Ziegler, D.; Oestrike, R.; Manar, P. (1989): Calorimetry of silicate melts at 1773 K: Measurement of enthalpies of fusion and of mixing in the systems diopside-anorthite-albite and anorthite-forsterite. Contrib. Mineral. Petrol. 101, 122-130, [doi:10.1007/BF00387206](http://dx.doi.org/10.1007/BF00387206)
# - Nguyen, J. H.; Holmes, N. C. (2004): Melting of iron at the physical conditions of the Earth's core. Nature 427, 339-342, [doi:10.1038/nature02248](http://dx.doi.org/10.1038/nature02248)
# - Nomura, R.; Hirose, K.; Uesugi, K.; Ohishi, Y.; Tsuchiyama, A.; Miyake, A.; Ueno, Y. (2014): Low core-mantle boundary temperature inferred from the solidus of pyrolite. Science 343(6170), 522-525, [doi:10.1126/science.1248186](http://dx.doi.org/10.1126/science.1248186)
# - Ohtani, E. (1987): Ultrahigh-pressure melting of a model chondritic mantle and pyrolite compositions. In: High-pressure Research in Mineral Physics, ed. by Manghnani, M. H. and Syono, Y., vol. 39 in Geophysical Monograph, Terrapub/American Geophysical Union, Tokyo/Washington, D.C., pp. 87-93, [doi:10.1029/GM039p0087](http://dx.doi.org/10.1029/GM039p0087)
# - Ohtani, E.; Kumazawa, M. (1981): Melting of forsterite Mg<sub>2</sub>SiO<sub>4</sub> up to 15 GPa. Phys. Earth Planet. Inter. 27, 32-38, [doi:10.1016/0031-9201(81)90084-4](http://dx.doi.org/10.1016/0031-9201(81)90084-4)
# - Ohtani, E.; Irifune, T.; Fujino, K. (1981): Fusion of pyrope at high pressures and rapid crystal growth from the pyrope melt. Nature 294, 62-64, [doi:10.1038/294062a0](http://dx.doi.org/10.1038/294062a0)
# - Ohtani, E.; Sawamoto, H. (1987): Melting experiment on a model chondritic mantle composition at 25 GPa. Geophys. Res. Lett. 14(7), 733-736, [doi:10.1029/GL014i007p00733](http://dx.doi.org/10.1029/GL014i007p00733)
# - Ono, S.; Oganov, A. R.; Brodholt, J. P.; Vočadlo, L.; Wood, I. G.; Lyakhov, A.; Glass, C. W.; Côté, A. S.; Price, G. D. (2008): High-pressure phase transformations of FeS: Novel phases at conditions of planetary cores. Earth Planet. Sci. Lett. 272(1-2), 481-487, [doi:10.1016/j.epsl.2008.05.017](http://dx.doi.org/10.1016/j.epsl.2008.05.017)
# - Palme, H.; O'Neill, H. St. C. (2014): Cosmochemical estimates of mantle composition. In: The Mantle and Core, ed. by Carlson, R. W., vol. 3 in Treatise on Geochemistry, Elsevier, pp. 1-39, [doi:10.1016/B978-0-08-095975-7.00201-1](http://dx.doi.org/10.1016/B978-0-08-095975-7.00201-1)
# - Pradhan, G. K.; Fiquet, G.; Siebert, J.; Auzende, A.-L.; Morard, G.; Antonangeli, D.; Garbarino, G. (2015): Melting of MORB at core-mantle boundary. Earth Planet. Sci. Lett. 431, 247-255, [doi:10.1016/j.epsl.2015.09.034](http://dx.doi.org/10.1016/j.epsl.2015.09.034)
# - Presnall, D. C.; Dixon, J. R.; O'Donnell, T. H.; Dixon, S. A. (1979): Generation of mid-ocean ridge tholeiites. J. Petrol. 20(1), 3-35, [doi:10.1093/petrology/20.1.3](http://dx.doi.org/10.1093/petrology/20.1.3)
# - Presnall, D. C.; Walter, M. J. (1993): Melting of forsterite, Mg<sub>2</sub>SiO<sub>4</sub>, from 9.7 to 16.5 GPa. J. Geophys. Res. 98(B11), 19777-19783, [doi:10.1029/93JB01007](http://dx.doi.org/10.1029/93JB01007)
# - Richet, P.; Bottinga, Y. (1984): Anorthite, andesine, wollastonite, diopside, cordierite and pyrope: thermodynamics of melting, glass transitions, and properties of the amorphous phases. Earth Planet. Sci. Lett. 67(3), 415-432, [doi:10.1016/0012-821X(84)90179-1](http://dx.doi.org/10.1016/0012-821X(84)90179-1)
# - Richet, P.; Leclerc, F.; Benoist, L. (1993): Melting of forsterite and spinel, with implications for the glass transition of Mg<sub>2</sub>SiO<sub>4</sub> liquid. Geophys. Res. Lett. 20(16), 1675-1678, [doi:10.1029/93GL01836](http://dx.doi.org/10.1029/93GL01836)
# - Ringwood, A. E.; Hibberson, W. (1990): The system Fe-FeO revisited. Phys. Chem. Min. 17(4), 313-319, [doi:10.1007/BF00200126](http://dx.doi.org/10.1007/BF00200126)
# - Ruedas, T.; Tackley, P. J.; Solomon, S. C. (2013): Thermal and compositional evolution of the martian mantle: Effects of phase transitions and melting. Phys. Earth Planet. Inter. 216, 32-58, [doi:10.1016/j.pepi.2012.12.002](http://dx.doi.org/10.1016/j.pepi.2012.12.002)
# - Ruedas, T.; Breuer, D. (2017): On the relative importance of thermal and chemical buoyancy in regular and impact-induced melting in a Mars-like planet. J. Geophys. Res. 122(7), 1554-1579, [doi:10.1002/2016JE005221](http://dx.doi.org/10.1002/2016JE005221)
# - Ruedas, T.; Breuer, D. (2018): "Isocrater" impacts: Conditions and mantle dynamical responses for different impactor types. Icarus 306, 94-115, [doi:10.1016/j.icarus.2018.02.005](http://dx.doi.org/10.1016/j.icarus.2018.02.005)
# - Ryzhenko, B.; Kennedy, G. C. (1973): The effect of pressure on the eutectic in the system Fe-FeS. Am. J. Sci. 273(9), 803-810, [doi:10.2475/ajs.273.9.803](http://dx.doi.org/10.2475/ajs.273.9.803)
# - Scarfe, C. M.; Takahashi, E. (1986): Melting of garnet peridotite to 13 GPa and the early history of the upper mantle. Nature 322, 354-356, [doi:10.1038/322354a0](http://dx.doi.org/10.1038/322354a0)
# - Schmerr, N. C.; Fei, Y.; Bertka, C. (2001): Extending the solidus for a model iron-rich martian mantle composition to 25 GPa. Lunar Planet. Sci. XXXII, [Abstract 1157](https://www.lpi.usra.edu/meetings/lpsc2001/pdf/1157.pdf)
# - Shen, G.; Lazor, P. (1995): Measurement of melting temperatures of some minerals under lower mantle pressures. J. Geophys. Res. 100(B9), 17699-17713, [doi:10.1029/95JB01864](http://dx.doi.org/10.1029/95JB01864)
# - Shen, G.; Mao, H.-k.; Hemley, R. J.; Duffy, T. S.; Rivers, M. L. (1998): Melting and crystal structure of iron at high pressures and temperatures. Geophys. Res. Lett. 25(3), 373-376, [doi:10.1029/97GL03776](http://dx.doi.org/10.1029/97GL03776)
# - Shen, G. Y.; Prakapenka, V. B.; River, M. L.; Sutton, S. R. (2004): Structure of liquid iron at pressures up to 58 GPa. Phys. Rev. Lett. 92(6), 185701, [doi:10.1103/PhysRevLett.92.185701](http://dx.doi.org/10.1103/PhysRevLett.92.185701)
# - Simon, F.; Glatzel, G. (1929): Bemerkungen zur Schmelzdruckkurve. Z. Anorg. Ang. Chem. 178(1), 309-316, [doi:10.1002/zaac.19291780123](http://dx.doi.org/10.1002/zaac.19291780123)
# - Stevenson, D. J. (1981): Models of the Earth's core. Science 214(4521), 611-619, [doi:10.1126/science.214.4521.611](http://dx.doi.org/10.1126/science.214.4521.611)
# - Stewart, A. J.; Schmidt, M. W.; van Westrenen, W.; Liebske, C. (2007): Mars: A new core-crystallization regime. Science 316(5829), 1323-1325, [doi:10.1126/science.1140549](http://dx.doi.org/10.1126/science.1140549)
# - Strong, H. M.; Tuft, R. E.; Hanneman, R. E. (1973): The iron fusion curve and $\gamma$-$\delta$-l triple point. Metall. Trans. 4(11), 2657-2661, [doi:10.1007/BF02644272](http://dx.doi.org/10.1007/BF02644272)
# - Swartzendruber, L. J. (1982): The Fe (iron) system. Bull. Alloy Phase Diagrams 3(2), 161-165, [doi:10.1007/BF02892374](http://dx.doi.org/10.1007/BF02892374)
# - Takahashi, E. (1983): [Melting of a Yamato L3 chondrite (Y-74191) up to 30 kbar.](http://id.nii.ac.jp/1291/00001558) Mem. Nat. Inst. Polar Res. Spec. Issue 30, 168-180
# - Takahashi, E. (1986): Melting of a dry peridotite KLB-1 up to 14 GPa: Implications on the origin of peridotitic upper mantle. J. Geophys. Res. 91(B9), 9367-9382, [doi:10.1029/JB091iB09p09367](http://dx.doi.org/10.1029/JB091iB09p09367)
# - Takahashi, E.; Scarfe, C. M. (1985): Melting of peridotite to 14 GPa and the genesis of komatiite. Nature 315, 566-568, [doi:10.1038/315566a0](http://dx.doi.org/10.1038/315566a0)
# - Takahashi, E.; Shimazaki, T.; Tsuzaki, Y.; Yoshida, H. (1993): Melting study of a peridotite KLB-1 to 6.5 GPa, and the origin of basaltic magmas. Phil. Trans. R. Soc. A 342(1663), 105-120, [doi:10.1098/rsta.1993.0008](http://dx.doi.org/10.1098/rsta.1993.0008)
# - Tateno, S.; Hirose, K.; Ohishi, Y.; Tatsumi, Y. (2010): The structure of iron in Earth's inner core. Science 330(6002), 359-361, [doi:10.1126/science.1194662](http://dx.doi.org/10.1126/science.1194662)
# - Taylor, S. R.; McLennan, S. M. (2009): Planetary Crusts. Cambridge University Press
# - Trønnes, R. G.; Frost, D. J. (2002): Peridotite melting and mineral-melt partitioning of major and minor elements at 22-24.5 GPa. Earth Planet. Sci. Lett. 197(1-2), 117-131, [doi:10.1016/S0012-821X(02)00466-1](http://dx.doi.org/10.1016/S0012-821X(02)00466-1)
# - Usselman, T. M. (1975): Experimental approach to the state of the core: part 1: the liquidus relations of the Fe-rich portion of the Fe-Ni-S system from 30 to 100 kbars. Am. J. Sci. 275(3), 278-290, [doi:10.2475/ajs.275.3.278](http://dx.doi.org/10.2475/ajs.275.3.278)
# - Wang, W.; Takahashi, E. (2000): Subsolidus and melting experiments of K-doped peridotite KLB-1 to 27 GPa: Its geophysical and geochemical implications. J. Geophys. Res. 105(B2), 2855-2868, [doi:10.1029/1999JB900366](http://dx.doi.org/10.1029/1999JB900366)
# - Williams, Q.; Jeanloz, R.; Bass, J.; Svendsen, B.; Ahrens, T. J. (1987): The melting curve of iron to 250 gigapascals: A constraint on the temperature at Earth's center. Science 236, 181-182, [doi:10.1126/science.236.4798.181](http://dx.doi.org/10.1126/science.236.4798.181)
# - Yasuda, A.; Fujii, T.; Kurita, K. (1994): Melting phase relations of an anhydrous mid-ocean ridge basalt from 3 to 20 GPa: Implications for the behaviour of subducted oceanic crust in the mantle. J. Geophys. Res. 99(B5), 9401-9414, [doi:10.1029/93JB03205](http://dx.doi.org/10.1029/93JB03205)
# - Yoder, Jr., H. S. (1952): Change of melting point of diopside with pressure. J. Geol. 60(4), 364-374, [doi:10.1086/625984](http://dx.doi.org/10.1086/625984)
# - Zerr, A.; Boehler, R. (1993): Melting of (Mg,Fe)SiO<sub>3</sub>-perovskite to 625 kilobars: Indication of high melting temperature in the lower mantle. Science 262, 553-555, [doi:10.1126/science.262.5133.553](http://dx.doi.org/10.1126/science.262.5133.553)
# - Zerr, A.; Serghiou, G.; Boehler, R. (1997): Melting of CaSiO<sub>3</sub> perovskite to 430 kbar and first *in-situ* measurements of lower mantle eutectic temperatures. Geophys. Res. Lett. 24(8), 909-912, [doi:10.1029/97GL00829](http://dx.doi.org/10.1029/97GL00829)
# - Zerr, A.; Diegeler, A.; Boehler, R. (1998): Solidus of Earth's Deep Mantle. Science 281, 243-246, [doi:10.1126/science.281.5374.243](http://dx.doi.org/10.1126/science.281.5374.243)
# - Zhang, D.; Jackson, J. M.; Zhao, J.; Sturhahn, W.; Alp, E. E.; Hu, M. Y.; Toellner, T. S.; Murphy, C. A.; Prakapenka, V. B. (2016): Temperature of Earth's core constrained from melting of Fe and Fe<sub>0.9</sub>Ni<sub>0.1</sub> at high pressures. Earth Planet. Sci. Lett. 447, 72-83, [doi:10.1016/j.epsl.2016.04.026](http://dx.doi.org/10.1016/j.epsl.2016.04.026)
# - Zhang, J.; Herzberg, C. (1994): [Melting of pyrope Mg<sub>3</sub>Al<sub>2</sub>Si<sub>3</sub>O<sub>12</sub>, at 7-16 GPa](http://www.minsocam.org/ammin/AM79/AM79_497.pdf). Amer. Mineral. 79(B9), 497-503
