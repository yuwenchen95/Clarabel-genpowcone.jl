
d = 10000

b = sqrt(5*d*d + 2*d + 1)
a = 3d − b + 1

t = [-sqrt(a/(2*(d+1))) ; (b-d+2)/(2*a*sqrt(d+1))*ones(d)]
t0 = [0.0; sqrt(1+1/d)*ones(d)]
resp = t[2] - t[1]
resd = d*t[2] + t[1]
println(resp," and ",resd)

res0p = t0[2] - t0[1]
res0d = d*t0[2] + t0[1]
println(res0p," and ",res0d)