
(define (problem footwear) (:domain footwear)
  (:objects
        beach0 - place
	beach1 - place
	beach2 - place
	beach3 - place
	beach4 - place
	foot1 - foot
	foot2 - foot
	forest0 - place
	forest1 - place
	forest2 - place
	forest3 - place
	forest4 - place
	gym0 - place
	gym1 - place
	gym2 - place
	gym3 - place
	gym4 - place
	home - place
	office0 - place
	office1 - place
	office2 - place
	office3 - place
	office4 - place
	shoe0 - shoe
	shoe1 - shoe
	shoe10 - shoe
	shoe11 - shoe
	shoe12 - shoe
	shoe13 - shoe
	shoe14 - shoe
	shoe15 - shoe
	shoe16 - shoe
	shoe17 - shoe
	shoe18 - shoe
	shoe19 - shoe
	shoe2 - shoe
	shoe20 - shoe
	shoe21 - shoe
	shoe22 - shoe
	shoe23 - shoe
	shoe24 - shoe
	shoe25 - shoe
	shoe26 - shoe
	shoe27 - shoe
	shoe28 - shoe
	shoe29 - shoe
	shoe3 - shoe
	shoe30 - shoe
	shoe31 - shoe
	shoe32 - shoe
	shoe33 - shoe
	shoe34 - shoe
	shoe35 - shoe
	shoe36 - shoe
	shoe37 - shoe
	shoe38 - shoe
	shoe39 - shoe
	shoe4 - shoe
	shoe5 - shoe
	shoe6 - shoe
	shoe7 - shoe
	shoe8 - shoe
	shoe9 - shoe
	sock0 - sock
	sock1 - sock
	sock10 - sock
	sock11 - sock
	sock12 - sock
	sock13 - sock
	sock14 - sock
	sock15 - sock
	sock16 - sock
	sock17 - sock
	sock18 - sock
	sock19 - sock
	sock2 - sock
	sock3 - sock
	sock4 - sock
	sock5 - sock
	sock6 - sock
	sock7 - sock
	sock8 - sock
	sock9 - sock
  )
  (:init 
	(at home)
	(beach beach0)
	(beach beach1)
	(beach beach2)
	(beach beach3)
	(beach beach4)
	(forest forest0)
	(forest forest1)
	(forest forest2)
	(forest forest3)
	(forest forest4)
	(gym gym0)
	(gym gym1)
	(gym gym2)
	(gym gym3)
	(gym gym4)
	(home home)
	(isbare foot1)
	(isbare foot2)
	(isblue sock0)
	(isblue sock10)
	(isblue sock11)
	(isblue sock12)
	(isblue sock13)
	(isblue sock18)
	(isblue sock19)
	(isblue sock1)
	(isblue sock4)
	(isblue sock5)
	(isblue sock6)
	(isblue sock7)
	(isboot shoe0)
	(isboot shoe10)
	(isboot shoe11)
	(isboot shoe14)
	(isboot shoe15)
	(isboot shoe1)
	(isboot shoe24)
	(isboot shoe25)
	(isboot shoe26)
	(isboot shoe27)
	(isdressshoe shoe12)
	(isdressshoe shoe13)
	(isdressshoe shoe16)
	(isdressshoe shoe17)
	(isdressshoe shoe28)
	(isdressshoe shoe29)
	(isdressshoe shoe36)
	(isdressshoe shoe37)
	(isplain sock0)
	(isplain sock10)
	(isplain sock11)
	(isplain sock12)
	(isplain sock13)
	(isplain sock14)
	(isplain sock15)
	(isplain sock16)
	(isplain sock17)
	(isplain sock1)
	(isplain sock4)
	(isplain sock5)
	(isplain sock8)
	(isplain sock9)
	(isred sock14)
	(isred sock15)
	(isred sock16)
	(isred sock17)
	(isred sock2)
	(isred sock3)
	(isred sock8)
	(isred sock9)
	(issandle shoe18)
	(issandle shoe19)
	(issandle shoe20)
	(issandle shoe21)
	(issandle shoe2)
	(issandle shoe38)
	(issandle shoe39)
	(issandle shoe3)
	(issandle shoe6)
	(issandle shoe7)
	(issandle shoe8)
	(issandle shoe9)
	(issneaker shoe22)
	(issneaker shoe23)
	(issneaker shoe30)
	(issneaker shoe31)
	(issneaker shoe32)
	(issneaker shoe33)
	(issneaker shoe34)
	(issneaker shoe35)
	(issneaker shoe4)
	(issneaker shoe5)
	(isstriped sock18)
	(isstriped sock19)
	(isstriped sock2)
	(isstriped sock3)
	(isstriped sock6)
	(isstriped sock7)
	(office office0)
	(office office1)
	(office office2)
	(office office3)
	(office office4)
	(shoefree shoe0)
	(shoefree shoe10)
	(shoefree shoe11)
	(shoefree shoe12)
	(shoefree shoe13)
	(shoefree shoe14)
	(shoefree shoe15)
	(shoefree shoe16)
	(shoefree shoe17)
	(shoefree shoe18)
	(shoefree shoe19)
	(shoefree shoe1)
	(shoefree shoe20)
	(shoefree shoe21)
	(shoefree shoe22)
	(shoefree shoe23)
	(shoefree shoe24)
	(shoefree shoe25)
	(shoefree shoe26)
	(shoefree shoe27)
	(shoefree shoe28)
	(shoefree shoe29)
	(shoefree shoe2)
	(shoefree shoe30)
	(shoefree shoe31)
	(shoefree shoe32)
	(shoefree shoe33)
	(shoefree shoe34)
	(shoefree shoe35)
	(shoefree shoe36)
	(shoefree shoe37)
	(shoefree shoe38)
	(shoefree shoe39)
	(shoefree shoe3)
	(shoefree shoe4)
	(shoefree shoe5)
	(shoefree shoe6)
	(shoefree shoe7)
	(shoefree shoe8)
	(shoefree shoe9)
	(shoeseq shoe0 shoe0)
	(shoeseq shoe10 shoe10)
	(shoeseq shoe11 shoe11)
	(shoeseq shoe12 shoe12)
	(shoeseq shoe13 shoe13)
	(shoeseq shoe14 shoe14)
	(shoeseq shoe15 shoe15)
	(shoeseq shoe16 shoe16)
	(shoeseq shoe17 shoe17)
	(shoeseq shoe18 shoe18)
	(shoeseq shoe19 shoe19)
	(shoeseq shoe1 shoe1)
	(shoeseq shoe20 shoe20)
	(shoeseq shoe21 shoe21)
	(shoeseq shoe22 shoe22)
	(shoeseq shoe23 shoe23)
	(shoeseq shoe24 shoe24)
	(shoeseq shoe25 shoe25)
	(shoeseq shoe26 shoe26)
	(shoeseq shoe27 shoe27)
	(shoeseq shoe28 shoe28)
	(shoeseq shoe29 shoe29)
	(shoeseq shoe2 shoe2)
	(shoeseq shoe30 shoe30)
	(shoeseq shoe31 shoe31)
	(shoeseq shoe32 shoe32)
	(shoeseq shoe33 shoe33)
	(shoeseq shoe34 shoe34)
	(shoeseq shoe35 shoe35)
	(shoeseq shoe36 shoe36)
	(shoeseq shoe37 shoe37)
	(shoeseq shoe38 shoe38)
	(shoeseq shoe39 shoe39)
	(shoeseq shoe3 shoe3)
	(shoeseq shoe4 shoe4)
	(shoeseq shoe5 shoe5)
	(shoeseq shoe6 shoe6)
	(shoeseq shoe7 shoe7)
	(shoeseq shoe8 shoe8)
	(shoeseq shoe9 shoe9)
	(sockfree sock0)
	(sockfree sock10)
	(sockfree sock11)
	(sockfree sock12)
	(sockfree sock13)
	(sockfree sock14)
	(sockfree sock15)
	(sockfree sock16)
	(sockfree sock17)
	(sockfree sock18)
	(sockfree sock19)
	(sockfree sock1)
	(sockfree sock2)
	(sockfree sock3)
	(sockfree sock4)
	(sockfree sock5)
	(sockfree sock6)
	(sockfree sock7)
	(sockfree sock8)
	(sockfree sock9)
	(socksmatch sock0 sock1)
	(socksmatch sock10 sock11)
	(socksmatch sock11 sock10)
	(socksmatch sock12 sock13)
	(socksmatch sock13 sock12)
	(socksmatch sock14 sock15)
	(socksmatch sock15 sock14)
	(socksmatch sock16 sock17)
	(socksmatch sock17 sock16)
	(socksmatch sock18 sock19)
	(socksmatch sock19 sock18)
	(socksmatch sock1 sock0)
	(socksmatch sock2 sock3)
	(socksmatch sock3 sock2)
	(socksmatch sock4 sock5)
	(socksmatch sock5 sock4)
	(socksmatch sock6 sock7)
	(socksmatch sock7 sock6)
	(socksmatch sock8 sock9)
	(socksmatch sock9 sock8)
  )
  (:goal (and
	(workedoutat gym2)
	(swamat beach2)))
)
