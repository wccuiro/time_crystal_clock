#!/usr/bin/env python3
import os, sys, struct
import numpy as np
import matplotlib.pyplot as plt
import zstandard as zstd

RECORD_SIZE = 1 + 2*8
UNPACKER    = struct.Struct('<B2d').unpack

def extract_time_jumps(path, desired_jump_type):
    tjs = []
    dctx = zstd.ZstdDecompressor()
    with open(path,'rb') as f, dctx.stream_reader(f) as r:
        while True:
            chunk = r.read(RECORD_SIZE)
            if len(chunk)<RECORD_SIZE: break
            jt, tj, *rest = UNPACKER(chunk)
            if jt==desired_jump_type:
                tjs.append(tj)
    return np.array(tjs)

def scan_Ms_streaming(folder, M_values, jump_type):
    stats = {m:{'sum':0.0,'sum_sq':0.0,'cnt':0} for m in M_values}
    for fn in os.listdir(folder):
        if not fn.endswith('.zst'): continue
        tjs = extract_time_jumps(os.path.join(folder,fn), jump_type)
        for m in M_values:
            num = len(tjs)//m
            if num<1: continue
            last = 0.0
            Ts = []
            for k in range(1, num+1):
                tk = tjs[k*m - 1]
                Ts.append(tk - last)
                last = tk
            T = np.array(Ts)
            s = stats[m]
            s['sum']    += T.sum()
            s['sum_sq'] += (T**2).sum()
            s['cnt']    += T.size

    Ms, As = [], []
    for m in M_values:
        s = stats[m]; N=s['cnt']
        if N>0:
            mu  = s['sum']/N
            var = (s['sum_sq']/N) - mu*mu
            A = mu*mu/var if var>0 else 0.0
        else:
            A = 0.0
        Ms.append(m); As.append(A)
    return np.array(Ms), np.array(As)

def find_optimal_M(folder, jump_type, M_max=1300, coarse_step=15, fine_win=50):
    Mc, Ac = scan_Ms_streaming(folder, list(range(coarse_step, M_max+1, coarse_step)), jump_type)
    M0 = Mc[np.nanargmax(Ac)]
    lo, hi = max(1,M0-fine_win), min(M_max,M0+fine_win)
    Mf, Af = scan_Ms_streaming(folder, list(range(lo, hi+1)), jump_type)
    M_star = Mf[np.nanargmax(Af)]
    return M_star, Af.max(), (Mc,Ac), (Mf,Af)

if __name__=="__main__":
    if len(sys.argv)!=3:
        print("Usage: python optimize_M.py ZST_FOLDER JUMP_TYPE"); sys.exit(1)
    folder, jtype = sys.argv[1], int(sys.argv[2])
    M_star,A_star,(Mc,Ac),(Mf,Af) = find_optimal_M(folder, jtype)
    print(f"Optimal M*: {M_star}   A = {A_star:.6f}")
    plt.figure(figsize=(8,5))
    plt.plot(Mc,Ac,'o-',label='Coarse scan')
    plt.plot(Mf,Af,'x--',label='Fine scan')
    plt.axvline(M_star,color='red',linestyle=':',label=f'M* = {M_star}')
    plt.xlabel('Threshold m')
    plt.ylabel(r'Accuracy $\langle T\rangle^2/\mathrm{Var}(T)$')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
