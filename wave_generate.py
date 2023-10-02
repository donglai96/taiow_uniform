import numpy as np
from scipy import integrate
import constants as cst
from taichiphysics import *
import random





class Waves_generate(object):
    """

    Because the operation of wave particle interaction is in taichi part,
    This code is just for initialization wave information

    """
    def __init__(self,frequencies, B0, ne, Bw, psi ,w_width, distribution ="Gaussian"):
        self.ne = ne
        self.B0 = B0 # B0 is background and Bw is wave
        self.psi = psi # wave normal angle, this should have size with nw, each wave 
                       # can have diiferent wave normal angle, but typically same.
                       # this should be rad
        self.ws =frequencies
        self.w_width = w_width
        self.nw = len(self.ws)
        print('total number of wave frequency is:',self.nw)

        if self.nw == 1:
            print('The input wave is monochromatic wave')
            print('The input frequency is :', self.ws)
            print('*************')
            print('Using the input magnetic field as field strength')
            self.Bwy = np.array([Bw])
        else:
            # using wave with a broadband
            if self.ws[-1] <= self.ws[0]:
                raise ValueError("The given frequencies should be from lower to higher!")
            
            Power_sum_square = 0
            self.Bwy = np.zeros(self.nw) 
            
            self.w_m = (self.ws[-1] + self.ws[0])/2
            print('medium frequency',self.w_m)
            
            # A Guassian distribution
            if distribution == "Gaussian":
                print("Using Guassian distribution of wave, it has some")
                for i in range(self.nw):
                    tmp = (self.ws[i] - self.w_m)/ self.w_width
                    self.Bwy[i] = np.exp(-tmp**2)
                    Power_sum_square += self.Bwy[i]
                Power_sum = np.sqrt(Power_sum_square)


                
                for i in range(self.nw):
                    self.Bwy[i] = np.sqrt(self.Bwy[i]) * Bw / Power_sum
            elif distribution == "Constant":
                print("Using constant distribution of wave")
                #self.Bwy = Bw / np.sqrt(self.nw)
                for i in range(self.nw):
                    self.Bwy[i] = Bw / np.sqrt(self.nw)
            else:
                raise ValueError("Unknown distribution")

# This is just parallel_wave
    def generate_parallel_wave(self,wce,wpe):
        
        # get wave phase, wave number, and the amplitude of E (calculated from B)
        #self.wce = gyrofrequency(cst.Charge,cst.Me, self.B0)
        #self.wpe = plasmafrequency(cst.Charge,cst.Me, self.ne)

        self.wce = wce
        self.wpe = wpe
        RR = 1 + self.wpe**2 / ((self.wce - self.ws) * self.ws) 
       
        self.k = self.ws * np.sqrt(RR)/cst.C 
        mu = cst.C * self.k / self.ws
        ExByp = self.direction / mu # Ex / By

        self.Ewx = self.Bwy * ExByp
        np.random.seed(66)

        self.phi0= np.random.rand(self.nw) * 2 * np.pi # 0 initial
        #self.phi0= 0


        # !! remember to get the random wave phase
        print("The information of wave:")
        print("gyrofrequency:", self.wce)
        print("plasmafrequency", self.wpe) 
        print("wave number k", self.k)
        print("wave frequency", self.ws)
        print("wave amplitude Bwy", self.Bwy)
        print("wave amplitude Ewx", self.Ewx)
        print("Initial phase", self.phi0)

    def generate_oblique_wave(self,wce,wpe):
        """Generate oblique wave based on STIX notation, notice this only return 
        to the value of B_xyz and E_xyz, the wave info need wave phase and the wave 
        phase is dtermined in the taiwave.py because it is related to the time and location
        of the particle which will pass from the main.py

         the wave normal vectoris in the x-z plane
         \vec k = k(sin(\psi),0,cos(\psi)) # kx, ky, kz

         Bw = { Bx cos(\phi), -Bysin(\phi), Bz cos(\phi) }
         Ew = { -Ex sin(\phi), -Eycos(\phi), -Ez sin(\phi) }

        """
        self.wce = wce
        self.wpe = wpe
        RR = 1 + self.wpe**2 / ((self.wce - self.ws) * self.ws) 
        LL = 1 - self.wpe**2 / ((self.wce + self.ws) * self.ws)
        PP = 1 - self.wpe**2 / (self.ws * self.ws)
        SS = 1/2 * (RR + LL)
        DD = 1/2 * (RR - LL)
        
        # calculate the k value
        sin = np.sin(self.psi)
        cos = np.cos(self.psi)
        rhs_up = RR * LL * sin**2 + PP * SS * (1 + cos**2) - ((RR * LL - PP *SS)**2 * sin**4 + 4 * PP**2 * DD**2 * cos**2)**0.5
        rhs_bot = 2 * (SS * sin**2 + PP*cos**2)
        self.k = (rhs_up / rhs_bot) ** 0.5 * self.ws /cst.C 
        

        # calculate the wave amplitude
        # Following https://npg.copernicus.org/articles/17/599/2010/
        # Notice this paper is not Gaussian unit
        eta = cst.C * self.k / self.ws

        Bx_By = DD * (eta**2 * sin**2 - PP) / (PP * (SS -eta**2))
        Bz_By = DD * sin * (PP - eta**2 * sin**2) / (PP * cos * (SS - eta**2))

        Ex_By = (PP - eta**2* sin**2)/(eta * PP * cos)
        Ey_By = DD * (PP - eta**2 * sin**2) / (eta*PP * cos * (eta**2 - SS ))  
        Ez_By = -1 * eta * sin / PP
        self.Bwx = self.Bwy * Bx_By
        self.Bwz = self.Bwy * Bz_By

        self.Ewx = self.Bwy * Ex_By
        self.Ewy = self.Bwy * Ey_By
        self.Ewz = self.Bwy * Ez_By

        np.random.seed(66)

        self.phi0= np.random.rand(self.nw) * 2 * np.pi # 0 initial
        print("The information of wave:")

        print("The init phase:",self.phi0)
        print("gyrofrequency:", self.wce)
        print("plasmafrequency", self.wpe) 
        print("wave number k", self.k)
        print("wave frequency", self.ws)
        
        print("wave amplitude Ewx", self.Ewx)
        print("wave amplitude Ewx", self.Ewy)
        print("wave amplitude Ewx", self.Ewz)
        print("wave amplitude Bwx", self.Bwx)
        print("wave amplitude Bwy", self.Bwy)
        print("wave amplitude Bwz", self.Bwz)
        print("Initial phase", self.phi0)
