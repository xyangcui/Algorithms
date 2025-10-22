
;--------------------------------------------------------------------
; monte carlo test 
; method 1.
; randomly shifting original dateset in recycle form. pre or back.
; This procedure maintains the autocorrelation.
; reference: Toride and Hakim (2021), Seo and Lee (2017)
;--------------------------------------------------------------------
; method 2.
; still, assume a martrix like (nevent, ndays).
; the first dimension is MJO event, the second one is MJO-related days.
; Then random select n days as the first day of each MJO event.
; This procedure can also maintain the autocorrelation.
; reference: Zheng et al. (2018) and so on. I don't know....
;--------------------------------------------------------------------
; method 3. (uh, monte carlo or bootstrap?)
; MJO pentad window.
; chunk MJO days in n days window,assume there are m chunks.
; Then randomly select m dates as the first day of each chunk.
; the same way, just do many times.
; codes wait for update.......
; reference: Jenny et al. (2019)
;--------------------------------------------------------------------

;--------------------------------------------------------------------
; random number.
;--------------------------------------------------------------------
function random_number(lb:integer,hb:integer,n:integer)
local lb,hb,n,rd
begin
  rseed1 = toint(systemfunc(" date +%s"))
  rseed2 = toint((12345l*rseed1)%2147483398l)
  random_setallseed(rseed1, rseed2)
  rd = floattointeger(random_uniform(lb,hb,n))
  return rd
end
;--------------------------------------------------------------------

;--------------------------------------------------------------------
; method 1.
;--------------------------------------------------------------------
function monte_carlo1(ind_time[*]:integer)
local ind_time,N,fi,nrand,n,indp,ind1,ind2,indp,indp1,ind_timet
begin
  N     = dimsizes(ind_time)
  fi    = N-1
  nrand = random_number(-1*N,N,1)
  if (nrand.lt.0) then
    n = abs(nrand)
    if (n.eq.N) then
      indp = ind_time
    else
      ind1 = ind_time(n:fi)
      ind2 = ind_time(0:n-1)
      indp = array_append_record(ind1, ind2, 0)
    end if
  else
    n = nrand
    if (n.eq.N) then
      indp = ind_time
    else
      ind_timet = ind_time(::-1)
      ind1  = ind_timet(n:fi)
      ind2  = ind_timet(0:n-1)
      indpt = array_append_record(ind1, ind2, 0)
      indp  = indpt(::-1)
    end if  
  end if

  return indp
end
;--------------------------------------------------------------------

;--------------------------------------------------------------------
; method 2.
; for convineces, the return value is time indices.
; ind_time: for example. ind1 has time values of general time.
; ind_time contains ind1's indices.
;--------------------------------------------------------------------
function monte_carlo2(ind_time[*]:integer,a[*]:integer,N:integer)
local ind_time,a,N,fi,n,nrand,x,x1
begin
  fi    = N-1
  n     = dimsizes(a)
  nrand = random_number(0,fi,n)
;; The first event.
  x = new(a(0),integer)
  x = 0
  do j = 0, a(0)-1
    x(j) = ind_time(nrand(0)) + j
  end do
;; following events.
  do i = 1,n-1
    x1 = new(a(i),integer)
    x1 = 0
    do j = 0, a(i)-1
      x1(j) = ind_time(nrand(i)) + j
    end do
    x := array_append_record(x,x1,0)
    delete(x1)
  end do
  return x
end
;--------------------------------------------------------------------

;--------------------------------------------------------------------
; method 3.
; for convineces, the return value is time indices.
; n: window.
; ndate: MJO dates.
; N: DJF dates.
;--------------------------------------------------------------------
function monte_carlo3(ind_time[*]:integer,N:integer,ndate:integer,nw)
local ind_time,N,fi,n,nrand,x,x1,ndate
begin
  n     = floattointeger(floor(1.*ndate/nw) + 1)
  nleft = floattointeger(ndate-(n-1)*nw)        ; the last chunk's days.
  fi    = N-1
  nrand = random_number(0,fi,n)
;; The first event.
  x = new(nw,integer)
  x = 0
  do j = 0, nw-1
    x(j) = ind_time(nrand(0)) + j
  end do
;; following events.
  do i = 1,n-2
    x1 = new(nw,integer)
    x1 = 0    
    do j = 0, nw-1
      x1(j) = ind_time(nrand(i)) + j
    end do
    x := array_append_record(x,x1,0)
    delete(x1)
  end do
;; the last chunk.
  x1 = new(nw,integer)
  x1 = 0 
  do j = 0, nw-1
    x1(j) = ind_time(nrand(i)) + j
  end do
  if (nleft.eq.0)then
    x := array_append_record(x,x1,0)
  else
    x2 = x1(0:nleft-1)
    x := array_append_record(x,x2,0)
  end if
  
  return x
end
;--------------------------------------------------------------------


;-------------------------------------------------
;; test
;-------------------------------------------------
;time = ispan(0, 100, 1)
;ntime = 1400
;ndate = 140
;n2  = 14
;a   = (/5,6,7,1,2,3,4,5,6/)

;b = monte_carlo1(time)
;b = monte_carlo2(time,a,ntime)
;b = monte_carlo3(time,ntime,ndate,n2)
;print(b)
;-------------------------------------------------