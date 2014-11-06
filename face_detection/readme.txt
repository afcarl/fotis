face_detect.py

	väljakutsumine (viimane argument on valikuline, nägude kataloog ei pea reaalselt eksisteerima):
		face_detect.py <piltide kataloog> <nägude kataloog> <näofaili suurus pikslites (ruudukujuline)>
		nt face_detect.py isikud isikud_faces 32


sort_faces.py
	tekitab kaustastruktuuri:
		<results folder>
			<Inimese 1 nimi>
				näopilt1
				näopilt2
			<Inimese 2 nimi>
				näopilt
			<Inimese 3 nimi>
				näopiltN

	väljakutsumine:
		sort_faces.py <CSV file> <faces folder> <results folder>
		nt sort_faces.py isikud.csv isikud_faces> isikud_results


prepare_batches.py
	sisendiks on sort_faces skriptile antud väljundkaust <results folder> ehk eeldab, et sisendiks on kaust, milles on
	isikunimedega kaustad ja nendes näofailid. skript muudab automaatselt näofaili suuruse sobivaks

	väljakutsumine (<face size> argument on valikuline):
		prepare_batches.py <folder with people-named-folders> <results folder> <batch size> <images per person> <face size>
		nt prepare_batches.py isikud_results isikud_batches 100 15