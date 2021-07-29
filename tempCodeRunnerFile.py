## Initial condition
ns = [6]
J = 1
h = np.linspace(0,1,num=11)
BC = 'PBC'
for i in range(len(ns)):
    n = ns[i]
    Szs = []
    hs = []
    for j in range(len(h)):
        w ,v ,arr = Spin05(n, J, h[j], BC)
        EVal_0 = w[0]
        EVal_1 = w[1]
        #EVec = v[:,0]
        Szs.append(EVal_1-EVal_0)
        hs.append(h[j])

    plt.plot(hs, Szs, '-o', markersize = 4, label = 'L=%d' %(n))

# print(Szs)
# print(hs)

plt.xlabel(r'h', fontsize=14)
plt.ylabel(r'$Sz(h)$', fontsize=14)
# plt.xlim(3,32)
# plt.ylim(0.001, 1)
# plt.xscale('log')
# plt.yscale('log')
plt.title(r'Sz vs h, J = %d' %(J), fontsize=12)
plt.legend(loc = 'best')
plt.savefig('/home/liusf/10902-Computational-Physics/2.pdf', format='pdf', dpi=4000)
# plt.show()