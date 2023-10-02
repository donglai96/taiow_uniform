import taichi as ti
import numpy as np
import constants as cst
import time
import sys
from taichiphysics import *
from res_energy import *
from particle import *
from wave_generate import *
from taiwave import Wave

if __name__ == '__main__':

    if len(sys.argv) > 1:
        print("The running id:", sys.argv[1:])
    else:
        print("No input id provided.")
    
    # read the input file
    id = sys.argv[1:][0]
    with open(id + '/input.txt', 'r') as f:
        lines = f.readlines()
    paras = {}
    for line in lines:
        if ':' in line:
            key, value = line.strip().split(': ')
            try:
                paras[key] = int(value)
            except ValueError:
                try:
                    paras[key] = float(value)
                except ValueError:
                    paras[key] = value


    # read the paras
    B0 = paras['B0']# Background magnetic field
    psi0 = np.deg2rad(paras['psi']) # wave normal angle
    wce = gyrofrequency(cst.Charge, cst.Me, B0)
    n0 = paras['n0']# density , I need to change this to wpe?
    wpe = plasmafrequency(cst.Charge,cst.Me,n0)


    print('******************')
    print('wce/wpe',{wce/wpe})
    #wpe = paras['wpe']
    wave_distribution = paras['wave_distribution']
    nres = paras['nres']
    Np = paras['Np']# particle numbers
    pitch_angle_degree = paras['pitch_angle_degree'] # to the background magnetic field!
    t_total_num = paras['t_total_num']
    record_num = paras['record_num']
    dt_num =paras['dt_num']
    w_res_num =paras['w_res_num']
    w_lc_num = paras['w_lc_num']
    w_uc_num = paras['w_uc_num']
    w_width_num = paras['w_width_num']
    nw = paras['nw']
    dz_num = paras['dz_num']
    Bw = paras['Bw']
    mass = cst.Me
    charge = cst.Charge * -1 # -1 is electron
    ########### init the taichi kernel
    ti.init(arch = ti.cpu,default_fp=ti.f64)

    # wave distribution

    print('Calculating resonance energy')
    #
    alpha = np.deg2rad(pitch_angle_degree) # pitch angle, notice the antidirection
    w = w_res_num * wce
    w_lc = w_lc_num * wce
    w_uc = w_uc_num * wce
    w_width = w_width_num * wce
    p0,k0 = get_resonance_whistler(w, wpe,wce,alpha,nres,psi0)

    print('momentum of resonating particles are:',p0)
    print('resonant wave number is :', k0)
    print('resonating frequency is (Hz) ', w/(2 * np.pi))
    print('resonating frequency is (rad/s) ', w)
    
    print('E0 is ', erg2ev(p2e(p0))/1000, ' keV')

    gamma = (1 + p0**2 / (cst.Me**2*cst.C**2))**0.5
    wce_rel = wce/gamma
    T_gyro = 2 * np.pi/ wce_rel
    dt = T_gyro * dt_num
    Nt = int(t_total_num/dt_num) 
    print(' time step is:', dt)
    print(' mass is ', mass)
    print(' charge is ', charge)
    print(' total time is', t_total_num * T_gyro)
    print(' total time step is ', Nt)
    particles = Particle.field(shape = (Np,))

    if nw > 1:
        iw_res = int((w - w_lc) / ((w_uc - w_lc) / (nw - 1)))
        dw = (w_uc - w_lc) / (nw - 1)
        w_lc_temp = w - iw_res * dw; 
        ws = np.array([i * dw for i in range(nw)] ) + w_lc_temp
        
    else:
        ws = np.array([w])
    print('resonance frequency:',w)
    print('ws')
    print(ws)
    k_res = k0
    wave_length = 2 * np.pi / k_res
    if Np > 1:
        dz = dz_num * wave_length/(Np - 1)
        
    else:
        dz = dz_num * wave_length

    dphi = 2 * np.pi / Np

    pperp = p0 * np.sin(alpha)

    px_numpy = np.zeros(Np)
    py_numpy = np.zeros(Np)
    pz_numpy = np.zeros(Np)

    for n in range(Np):
        phi = dphi * n
        px_numpy[n] = pperp * np.cos(phi)
        py_numpy[n] = pperp * np.sin(phi)
        pz_numpy[n] = p0 * np.cos(alpha)

    px_init = ti.field(dtype = ti.f64,shape = (Np,))
    py_init= ti.field(dtype = ti.f64,shape = (Np,))
    pz_init= ti.field(dtype = ti.f64,shape = (Np,))

    px_init.from_numpy(px_numpy)
    py_init.from_numpy(py_numpy)
    pz_init.from_numpy(pz_numpy)


    # init the wave
    waves_init = Waves_generate(ws, B0, n0, Bw, psi0 ,w_width, distribution =wave_distribution)
    waves_init.generate_oblique_wave(wce,wpe)

    # init quantities in taichi scope

    ws_taichi = ti.field(dtype = ti.f64,shape = (nw,))
    phi0_taichi = ti.field(dtype = ti.f64,shape = (nw,))
    Ewx_taichi = ti.field(dtype = ti.f64,shape = (nw,))
    Ewy_taichi = ti.field(dtype = ti.f64,shape = (nw,))
    Ewz_taichi = ti.field(dtype = ti.f64,shape = (nw,))

    Bwx_taichi = ti.field(dtype = ti.f64,shape = (nw,))
    Bwy_taichi = ti.field(dtype = ti.f64,shape = (nw,))
    Bwz_taichi = ti.field(dtype = ti.f64,shape = (nw,))
    k_taichi = ti.field(dtype = ti.f64,shape = (nw,))
    
    ws_taichi.from_numpy(waves_init.ws)
    phi0_taichi.from_numpy(waves_init.phi0)
    Ewx_taichi.from_numpy(waves_init.Ewx)
    Ewy_taichi.from_numpy(waves_init.Ewy)
    Ewz_taichi.from_numpy(waves_init.Ewz)

    Bwx_taichi.from_numpy(waves_init.Bwx)
    Bwy_taichi.from_numpy(waves_init.Bwy)
    Bwz_taichi.from_numpy(waves_init.Bwz)
    k_taichi.from_numpy(waves_init.k)




    waves = Wave.field(shape = (nw,))
    # ts  =  ti.field(ti.f64, shape=())
    dt_taichi = ti.field(ti.f64, shape=())
    dt_taichi[None] = dt


    print('Record num is', Nt//record_num)
    if Nt%record_num > 0:
        
        p_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num+1, Np))
        E_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num+1, Np))

        B_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num+1, Np))

        r_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num+1, Np))
        phi_record_taichi = ti.field(dtype = ti.f64,shape = (Nt//record_num+1, Np))
    else:
        
        p_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num, Np))
        E_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num, Np))

        B_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num, Np))

        r_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num, Np))
        phi_record_taichi = ti.field(dtype = ti.f64,shape = (Nt//record_num, Np))

    # init function
    @ti.kernel
    def init():
        for n in range(Np):
            particles[n].initParticles(mass, charge)
            particles[n].initPos(0.0,0.0,dz * n)
            particles[n].initMomentum(px_init[n], py_init[n],pz_init[n])
        for m in range(nw):
            waves[m].initialize(ws_taichi[m],phi0_taichi[m],Ewx_taichi[m],
                                Ewy_taichi[m],Ewz_taichi[m],Bwx_taichi[m],Bwy_taichi[m],Bwz_taichi[m],k_taichi[m],psi0)
        B =ti.Vector([0.0,0.0,B0])
        E = ti.Vector([0.0,0.0,0.0])
        for m in range(nw):
            
            waves[m].get_wavefield( particles[n].t,particles[n].r)
            B += waves[m].Bw
            E += waves[m].Ew
        # print('E0 at t0',E)
        # print('B0 at t0',B)
        for n in range(Np):
            particles[n].boris_push(-dt_taichi[None]/2,E,B)
            
    @ti.kernel
    def simulate():
        # one step

        for n in range(Np):
            # get field
            B =ti.Vector([0.0,0.0,B0])
            E = ti.Vector([0.0,0.0,0.0])
            for m in range(nw):
                
                waves[m].get_wavefield(particles[n].t,particles[n].r )
                B += waves[m].Bw
                E += waves[m].Ew
            
            particles[n].t += dt_taichi[None] 
            particles[n].leap_frog(dt_taichi[None],E,B)
            


    @ti.kernel
    def simulate_t():
        for n in range(Np): # This will be Parallelized
            #particles
            for tt in range(Nt): # This will be Serialized
                #time
                B =ti.Vector([0.0,0.0,B0])
                E = ti.Vector([0.0,0.0,0.0])
                # rrr = particles[n].r #?
                # ttt = particles[n].t #?
                #print('********', particles[n].r)
                for m in range(nw):
                    #print('ttt',ttt,n, particles[n].t)
                    #print('rrr',rrr,n, particles[n].r)
                    #print('inside', particles[n].r)
                    waves[m].get_wavefield( particles[n].t,particles[n].r) # get Bw and Ew
                    # if use particles[n].r and t would be wrong
                    
                    B += waves[m].Bw
                    E += waves[m].Ew
                #print('Magnetic field!!',B)
                # print('t', particles[n].t)
                # print('E',E)
                # print('B',B)
                particles[n].t += dt_taichi[None] 
                particles[n].leap_frog(dt_taichi[None],E,B) # change nth particle's p and r
                particles[n].Ep = E
                particles[n].Bp = B
                #print('magnetic field!!',particles[n].Bp)
                #phib = ti.atan2(B[1],B[0])
                phip = ti.atan2(particles[n].p[1],particles[n].p[0])
                particles[n].phi = phip
                if (particles[n].phi < 0):
                    particles[n].phi += 2*ti.math.pi
                # save particle info
                if tt%record_num ==0:
                    #print('tt',tt)
                    #print('tt//record_num',tt//record_num)
                    p_record_taichi[tt//record_num, n] = particles[n].p
                    r_record_taichi[tt//record_num, n] = particles[n].r
                    phi_record_taichi[tt//record_num, n] = particles[n].phi
                    E_record_taichi[tt//record_num, n] = particles[n].Ep
                    B_record_taichi[tt//record_num, n] = particles[n].Bp
    ###################################################
    # Begin of the main loop
    start_time = time.time()
    init()
    #print(particles[1].r)
    simulate_t()
    


    print('finished')
    print('The init p is',p0)
    # End of the main loop
    ###################################################
    time_used = time.time() - start_time
    print("--- %s seconds ---" % (time_used))
    
    ###################################################

    p_results = p_record_taichi.to_numpy()
    r_results = r_record_taichi.to_numpy()
    Ep_results = E_record_taichi.to_numpy()
    Bp_results = B_record_taichi.to_numpy()
    phi_results = phi_record_taichi.to_numpy()



    with open(id + '/' + 'p_r_phi.npy','wb') as f:
        np.save(f,p_results)
        np.save(f,r_results)
        np.save(f,phi_results)


    with open(id + '/' + 'E_B.npy','wb') as f:
        np.save(f,Ep_results)
        np.save(f,Bp_results)
        #np.save(f,phi_results)
    # save the output info for checking
    with open(id + '/' + 'output.txt', 'w') as f:
        f.write(f"The resonating frequency is {w/(2 * np.pi)} (Hz)\n")

        f.write(f"momentum of resonating particles are: {p0}\n")
        f.write(f"E0 is  {erg2ev(p2e(p0))/1000} keV\n")
        f.write(f"--- %s seconds ---" % (time_used))