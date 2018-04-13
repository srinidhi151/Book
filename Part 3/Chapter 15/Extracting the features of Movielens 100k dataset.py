def parse(filename="ratings.csv", has_header=True, quiet=False):
   """
   Input file name.
   Required format:
      userId,movieId,rating,...
      int, int, float
   Set has_header to False if the file does not have a header.
   Set quiet to False to print stages.
   Returns a matrix with rows representing users and columns representing movies and mapping of movieId: column index.
   Discards all movies that do not have any rating.
   """
   R = {}
   movies = set()
   if not quiet:
      print("Reading...")
   with open(filename) as f:
      if has_header:
         if not quiet:
            print("Skipping header...")
         f.readline()
      for line in f:
         u, m, r = (lambda x, y, z: (int(x), int(y), float(z)))(*line.split(',')[:3])
         if u not in R:
            R[u] = {}
         R[u][m] = r
         movies.add(m)
   if not quiet:
      print("Read %d users and %d movies" % (len(R), len(movies)))
   movies = {i: j for i, j in zip(sorted(movies), range(len(movies)))}
   if not quiet:
      print("Generating matrix...")
   rat = [[0.0] * len(movies) for _ in range(len(R))]
   for u in R:
      for m in R[u]:
         rat[u - 1][movies[m]] = R[u][m]
   R = rat

   if not quiet:
      print("Done...")
      print(sum(map(len, R)), "values in matrix in", len(R), "rows")
   return R
