# -*- coding: utf-8 -*-
"""
Created on Thu June 27 13:00:58 2018

@author: William J. Trenberth
"""

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import scipy.fftpack as fft

def main():
    #The main parameters
    N = 2**8
    dt = 0.001
    eps = 0.0001
    Nt = int(10*dt*eps*N**4)
    
    #discretising the unit interval.
    x = np.linspace(0,1,N)
    x = np.delete(x,-1)
    
    #The inital condition.
    #u0 = random_initial_data(N,3.5)
    u0 = 1/np.cosh(np.sqrt(1/(12*eps))*(x - 0.8))**2

    #Animating the solution. See the Animate class below.    
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(-3,3))
    line, = ax.plot([], [], lw=2)
    animate = Animate(line,dt,Nt,u0,eps)  
    anim = animation.FuncAnimation(fig, animate, frames=8000, interval=20)
    
    #Saving the animation
    anim.save("KdV6.mp4")
    
def KdV_solver(t, Nt, u0, eps):
    '''A function to calculate, using an explicit, three point finite difference 
    method,  the solution to the KdV equation
    $$\partial_tu +\partial_x^3u + \eps u\partial_xu = 0$$ with inital data
    $u(x,0)=u_0$. Here t is the time you are finding the solution at, Nt is 
    the number of timesteps to get there and eps = \eps is the coefficent of 
    the nonlinearity. If eps is positive the equation is defocusing. If eps is
    negative the equation is focusing.
    '''
    
    #Calculating Number of spatial/temporal points and spatial/temporal step size.
    Nx = len(u0)
    dx = 1.0/Nx
    dt = 1.0*t/Nt
    
    #Initalialsing the first 2 steps in the FD method. This is a three point method
    #so two inital steps are needed to kick things off.
    v = np.zeros(Nx+4)
    u = np.zeros(Nx+4)     
    
    '''As this is a three step method with periodic boundary conditions I pad 
    the begining and end with appropriate values to simplfy things.
    '''
    #The first step
    v[2:Nx+2] = u0
    v[0:2] = u0[Nx-2:Nx]
    v[Nx+2:Nx+4] = u0[0:2]
    
    #The second step.
    u[2:Nx+2] = v[2:Nx+2] + (dt/dx)*(v[3:Nx+3] + v[2:Nx+2] + v[1:Nx+1])*(v[3:Nx+3] - v[1:Nx+1]) 
    + eps*(dt/dx**3)*(v[4:Nx+4] - 2*v[3:Nx+3] + 2*v[1:Nx+1]- v[0:Nx])
    u[0:2] = u[Nx:Nx+2]
    u[Nx+2:Nx+4] = u[2:4]
    
    '''The main loop in the finite difference method. I have partially 
    vectorised the code to increase the efficency. I couldn't figure out how to
    vectorie it in time. Doing this would massively increase the efficeny of
    this code.
    '''
    for Ni in range(0,Nt+1):
        (u[2:Nx+2], v) = (v[2:Nx+2] + (dt/dx)*(u[3:Nx+3] + u[2:Nx+2] + u[1:Nx+1])*(u[3:Nx+3] - u[1:Nx+1])
                            + eps*(dt/dx**3)*(u[4:Nx+4] - 2*u[3:Nx+3] + 2*u[1:Nx+1]- u[0:Nx]), u)
        u[0:2] = u[Nx:Nx+2]
        u[Nx+2:Nx+4] = u[2:4]  
        
    return u[2:Nx+2]

class Animate():
    '''Used to define an animate object used to animate the solution. These
    objects store the current u0 value, after KdV_solver is called, in a way 
    sort of replicating a static function variable. If you want to c
    compute the solution at time t_2 and have the value at t_1 you can go 
    from t_1 to t_2 instead of 0 to t_2 which would be inefficent.
    '''
    def __init__(self,line, dt, Nt, u0, eps):
        self.line = line
        self.dt = dt
        self.Nt = Nt
        self.u = u0
        self.eps = eps
        self.x = np.linspace(0,1,len(u0))
        
    def __call__(self,i):
        self.u = KdV_solver(self.dt, self.Nt, self.u, self.eps)
        self.line.set_data(self.x, self.u)
        return self.line
    
def random_initial_data(N,s):
    '''Using the DFT,  randomly generates a funcion of the form 
    $\sum\limits_{k=-N/2-1}^{N/2} \frac{g_k}{\langle k\rangle^s}e^{k\pi x}$
    where $g_k$ is a sequence of independent identically distributed 
    random variables and $\overline{g_k} = g_{-k} $.
    '''
    k = np.arange(1,N +1)
    Ff = np.random.randn(N)/((k**2 + 1)**(s/2))
    f = fft.irfft(Ff)
    f = 1000*np.delete(f,-1)
    
    return f

if __name__ == "__main__": main()
    
    

                                   
       
        
