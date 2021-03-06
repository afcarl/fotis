select count(1) from pildid -- 63323
select count(1) from fotod -- 63323

select count(1) from fotod join pildid using (pilt_id) -- 63323
-- pildid ja fotod on 1-1 seoses

select * from pildid where failinimi like 'efa0001_000_0000000_121783_f.jpg'

-- pilte väikeste failidena 9722
-- pilte suurte failidena 321906

select count(1) from isikud -- 31697

select count(1) from isikud join fotod using (foto_id) -- 31697
-- selliseid isikuid pole, kelle fotot meil pole

select count(1) from fotod left join isikud using (foto_id) where isikud.foto_id is null
-- fotosid ilma isikuteta: 42122

select count(1) from (select distinct foto_id from isikud) a
-- 21201 pilti on isikutega

select 21201+42122 -- 63323, kõik klapib

-- suurimate isikute arvudega pildid:
select foto_id, count(1) from isikud group by foto_id order by 2 desc limit 10

-- suurima esinemissagedusega isikud
select substr(eesnimi,1,1) || '. ' || perenimi, count(1) from isikud group by 1 order by 2 desc limit 10

-- suuremad kaustad
select kaust, count(1) from pildid group by 1 order by 2 desc

-- faile mis pole kahes suuremas kaustas on 1080, täpselt sama arv faile on lisa.zip failis
select * from pildid where kaust not in ('fotod/efa/efa0001/000/0000000/', 'fotod/efa/efa0001/001/0000000/')

-- piltide arv, kus esineb ainult üks isik
select count(1) from (select 1 from isikud group by foto_id having count(1) = 1) a
-- 14749

-- isikuid üheisikulistel piltidel
select count(distinct substr(eesnimi,1,1) || '. ' || perenimi) from
(select foto_id from isikud group by foto_id having count(1) = 1) a
join isikud using (foto_id)
-- 7076

-- millised isikud on kõige rohkem üksikuna pildi peal
select substr(eesnimi,1,1) || '. ' || perenimi, count(1)
from (select foto_id from isikud group by foto_id having count(1) = 1) f
join isikud i using (foto_id)
group by 1
having count(1) >= 20
order by 2 desc
-- 53 kirjet

-- üheisikuliste fotode kirjeldused
select foto_id, eesnimi, perenimi, rea_nr, nr_reas, sisu from
(select foto_id from isikud group by foto_id having count(1) = 1) a
join fotod using (foto_id)
join isikud using (foto_id)

-- üheisikuliste fotode failid
select foto_id, eesnimi, perenimi, sisu, failinimi from
(select foto_id from isikud group by foto_id having count(1) = 1) a
join fotod using (foto_id)
join pildid using (pilt_id)
join isikud using (foto_id)
where eesnimi like  'L%' and perenimi = 'Meri'

-- ainult üheisikuliste isikute pildid
create table tmp_failid as
select
	case p.kaust
    	when 'fotod/efa/efa0001/000/0000000/' then '000/0000000/'
        when 'fotod/efa/efa0001/001/0000000/' then '001/0000000/'
        else 'lisa/'
    end || p.failinimi as failinimi,
    substr(eesnimi,1,1) || '. ' || perenimi as nimi,
    sisu
from (select foto_id from isikud group by foto_id having count(1) = 1) a
join fotod f using (foto_id)
join pildid p using (pilt_id)
join isikud i using (foto_id)
-- ainult need isikud, kelle kohta on rohkem kui 20 pilti
where substr(eesnimi,1,1) || '. ' || perenimi in (
  select substr(eesnimi,1,1) || '. ' || perenimi
  from (select foto_id from isikud group by foto_id having count(1) = 1) f
  join isikud i using (foto_id)
  group by 1
  having count(1) >= 20
)

-- kontroll, et saame sama palju isikuid
select nimi, count(1) from tmp_failid group by 1 order by 2 desc

select * from tmp_failid order by nimi

-- piltide arvud grupeerituna kaustade kaupa
select
	case kaust
    	when 'fotod/efa/efa0001/000/0000000/' then '000/0000000/'
        when 'fotod/efa/efa0001/001/0000000/' then '001/0000000/'
        else 'lisa'
    end, count(1)
from pildid
group by 1

select
	case p.kaust
    	when 'fotod/efa/efa0001/000/0000000/' then '000/0000000/'
        when 'fotod/efa/efa0001/001/0000000/' then '001/0000000/'
        else 'lisa/'
    end || p.failinimi as failinimi,
	substr(eesnimi,1,1) || '. ' || perenimi as nimi,
    sisu
from isikud
join pildid using (foto_id)
join fotod using (pilt_id)
