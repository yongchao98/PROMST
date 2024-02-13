


(define (problem logistics-c21-s2-p23-a51)
(:domain logistics-strips)
(:objects a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26 a27 a28 a29 a30 a31 a32 a33 a34 a35 a36 a37 a38 a39 a40 a41 a42 a43 a44 a45 a46 a47 a48 a49 a50 
          c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20 
          t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12 t13 t14 t15 t16 t17 t18 t19 t20 
          l00 l01 l10 l11 l20 l21 l30 l31 l40 l41 l50 l51 l60 l61 l70 l71 l80 l81 l90 l91 l100 l101 l110 l111 l120 l121 l130 l131 l140 l141 l150 l151 l160 l161 l170 l171 l180 l181 l190 l191 l200 l201 
          p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 
)
(:init
(AIRPLANE a0)
(AIRPLANE a1)
(AIRPLANE a2)
(AIRPLANE a3)
(AIRPLANE a4)
(AIRPLANE a5)
(AIRPLANE a6)
(AIRPLANE a7)
(AIRPLANE a8)
(AIRPLANE a9)
(AIRPLANE a10)
(AIRPLANE a11)
(AIRPLANE a12)
(AIRPLANE a13)
(AIRPLANE a14)
(AIRPLANE a15)
(AIRPLANE a16)
(AIRPLANE a17)
(AIRPLANE a18)
(AIRPLANE a19)
(AIRPLANE a20)
(AIRPLANE a21)
(AIRPLANE a22)
(AIRPLANE a23)
(AIRPLANE a24)
(AIRPLANE a25)
(AIRPLANE a26)
(AIRPLANE a27)
(AIRPLANE a28)
(AIRPLANE a29)
(AIRPLANE a30)
(AIRPLANE a31)
(AIRPLANE a32)
(AIRPLANE a33)
(AIRPLANE a34)
(AIRPLANE a35)
(AIRPLANE a36)
(AIRPLANE a37)
(AIRPLANE a38)
(AIRPLANE a39)
(AIRPLANE a40)
(AIRPLANE a41)
(AIRPLANE a42)
(AIRPLANE a43)
(AIRPLANE a44)
(AIRPLANE a45)
(AIRPLANE a46)
(AIRPLANE a47)
(AIRPLANE a48)
(AIRPLANE a49)
(AIRPLANE a50)
(CITY c0)
(CITY c1)
(CITY c2)
(CITY c3)
(CITY c4)
(CITY c5)
(CITY c6)
(CITY c7)
(CITY c8)
(CITY c9)
(CITY c10)
(CITY c11)
(CITY c12)
(CITY c13)
(CITY c14)
(CITY c15)
(CITY c16)
(CITY c17)
(CITY c18)
(CITY c19)
(CITY c20)
(TRUCK t0)
(TRUCK t1)
(TRUCK t2)
(TRUCK t3)
(TRUCK t4)
(TRUCK t5)
(TRUCK t6)
(TRUCK t7)
(TRUCK t8)
(TRUCK t9)
(TRUCK t10)
(TRUCK t11)
(TRUCK t12)
(TRUCK t13)
(TRUCK t14)
(TRUCK t15)
(TRUCK t16)
(TRUCK t17)
(TRUCK t18)
(TRUCK t19)
(TRUCK t20)
(LOCATION l00)
(in-city  l00 c0)
(LOCATION l01)
(in-city  l01 c0)
(LOCATION l10)
(in-city  l10 c1)
(LOCATION l11)
(in-city  l11 c1)
(LOCATION l20)
(in-city  l20 c2)
(LOCATION l21)
(in-city  l21 c2)
(LOCATION l30)
(in-city  l30 c3)
(LOCATION l31)
(in-city  l31 c3)
(LOCATION l40)
(in-city  l40 c4)
(LOCATION l41)
(in-city  l41 c4)
(LOCATION l50)
(in-city  l50 c5)
(LOCATION l51)
(in-city  l51 c5)
(LOCATION l60)
(in-city  l60 c6)
(LOCATION l61)
(in-city  l61 c6)
(LOCATION l70)
(in-city  l70 c7)
(LOCATION l71)
(in-city  l71 c7)
(LOCATION l80)
(in-city  l80 c8)
(LOCATION l81)
(in-city  l81 c8)
(LOCATION l90)
(in-city  l90 c9)
(LOCATION l91)
(in-city  l91 c9)
(LOCATION l100)
(in-city  l100 c10)
(LOCATION l101)
(in-city  l101 c10)
(LOCATION l110)
(in-city  l110 c11)
(LOCATION l111)
(in-city  l111 c11)
(LOCATION l120)
(in-city  l120 c12)
(LOCATION l121)
(in-city  l121 c12)
(LOCATION l130)
(in-city  l130 c13)
(LOCATION l131)
(in-city  l131 c13)
(LOCATION l140)
(in-city  l140 c14)
(LOCATION l141)
(in-city  l141 c14)
(LOCATION l150)
(in-city  l150 c15)
(LOCATION l151)
(in-city  l151 c15)
(LOCATION l160)
(in-city  l160 c16)
(LOCATION l161)
(in-city  l161 c16)
(LOCATION l170)
(in-city  l170 c17)
(LOCATION l171)
(in-city  l171 c17)
(LOCATION l180)
(in-city  l180 c18)
(LOCATION l181)
(in-city  l181 c18)
(LOCATION l190)
(in-city  l190 c19)
(LOCATION l191)
(in-city  l191 c19)
(LOCATION l200)
(in-city  l200 c20)
(LOCATION l201)
(in-city  l201 c20)
(AIRPORT l00)
(AIRPORT l10)
(AIRPORT l20)
(AIRPORT l30)
(AIRPORT l40)
(AIRPORT l50)
(AIRPORT l60)
(AIRPORT l70)
(AIRPORT l80)
(AIRPORT l90)
(AIRPORT l100)
(AIRPORT l110)
(AIRPORT l120)
(AIRPORT l130)
(AIRPORT l140)
(AIRPORT l150)
(AIRPORT l160)
(AIRPORT l170)
(AIRPORT l180)
(AIRPORT l190)
(AIRPORT l200)
(OBJ p0)
(OBJ p1)
(OBJ p2)
(OBJ p3)
(OBJ p4)
(OBJ p5)
(OBJ p6)
(OBJ p7)
(OBJ p8)
(OBJ p9)
(OBJ p10)
(OBJ p11)
(OBJ p12)
(OBJ p13)
(OBJ p14)
(OBJ p15)
(OBJ p16)
(OBJ p17)
(OBJ p18)
(OBJ p19)
(OBJ p20)
(OBJ p21)
(OBJ p22)
(at t0 l01)
(at t1 l10)
(at t2 l21)
(at t3 l31)
(at t4 l41)
(at t5 l50)
(at t6 l61)
(at t7 l71)
(at t8 l81)
(at t9 l91)
(at t10 l100)
(at t11 l111)
(at t12 l120)
(at t13 l130)
(at t14 l140)
(at t15 l151)
(at t16 l161)
(at t17 l171)
(at t18 l180)
(at t19 l190)
(at t20 l200)
(at p0 l41)
(at p1 l80)
(at p2 l190)
(at p3 l40)
(at p4 l11)
(at p5 l01)
(at p6 l160)
(at p7 l131)
(at p8 l150)
(at p9 l51)
(at p10 l141)
(at p11 l160)
(at p12 l121)
(at p13 l41)
(at p14 l200)
(at p15 l151)
(at p16 l31)
(at p17 l00)
(at p18 l181)
(at p19 l141)
(at p20 l51)
(at p21 l60)
(at p22 l171)
(at a0 l60)
(at a1 l80)
(at a2 l140)
(at a3 l190)
(at a4 l30)
(at a5 l100)
(at a6 l80)
(at a7 l10)
(at a8 l10)
(at a9 l140)
(at a10 l40)
(at a11 l50)
(at a12 l50)
(at a13 l50)
(at a14 l50)
(at a15 l130)
(at a16 l50)
(at a17 l140)
(at a18 l190)
(at a19 l110)
(at a20 l100)
(at a21 l140)
(at a22 l150)
(at a23 l80)
(at a24 l100)
(at a25 l90)
(at a26 l40)
(at a27 l110)
(at a28 l10)
(at a29 l140)
(at a30 l40)
(at a31 l50)
(at a32 l200)
(at a33 l160)
(at a34 l10)
(at a35 l30)
(at a36 l50)
(at a37 l70)
(at a38 l20)
(at a39 l40)
(at a40 l00)
(at a41 l50)
(at a42 l100)
(at a43 l30)
(at a44 l80)
(at a45 l130)
(at a46 l160)
(at a47 l140)
(at a48 l50)
(at a49 l130)
(at a50 l20)
)
(:goal
(and
(at p0 l10)
(at p1 l201)
(at p2 l91)
(at p3 l201)
(at p4 l20)
(at p5 l180)
(at p6 l201)
(at p7 l160)
(at p8 l181)
(at p9 l150)
(at p10 l190)
(at p11 l190)
(at p12 l81)
(at p13 l61)
(at p14 l10)
(at p15 l90)
(at p16 l91)
(at p17 l61)
(at p18 l171)
(at p19 l01)
(at p20 l150)
(at p21 l10)
(at p22 l100)
)
)
)


