create database indian_celebraties ;
use  indian_celebraties ;

create table celebraty_information
( ID int auto_increment PRIMARY KEY,
  celebraty_Name varchar(50) ,
  Gender char(1),
  profession varchar(10),
  best_movie varchar(50),
  image text,
  details text
);
insert into celebraty_information
values("Vijay", "M", "Actor", "Soccer", "vijay.jpg", "bad boy");

select * from celebraty_information ;
#error code : 1175 : 
SET SQL_SAFE_UPDATES = 0 ;
delete from celebraty_information ;

drop table celebraty_information ;
