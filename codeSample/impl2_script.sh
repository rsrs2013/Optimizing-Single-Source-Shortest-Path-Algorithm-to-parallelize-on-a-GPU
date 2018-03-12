echo "Configuration $1,$2 : Work-effecient Parallel Method No Shared Memory Outcore"
echo "roadNet-CA Graph Results"
./sssp --input ./Graphs/roadNet-CA.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist input
echo "***********Source***************"
./sssp --input ./Graphs/roadNet-CA.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist source
echo "***********Destination***************"
./sssp --input ./Graphs/roadNet-CA.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist destination

echo "msdoor"
./sssp --input ./Graphs/msdoor.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist input
echo "***********Source***************"
./sssp --input ./Graphs/msdoor.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist source
echo "***********Destination***************"
./sssp --input ./Graphs/msdoor.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist destination

echo "road-CAL.txt"
./sssp --input ./Graphs/road-CAL.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist input
echo "***********Source***************"
./sssp --input ./Graphs/road-CAL.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist source
echo "***********Destination***************"
./sssp --input ./Graphs/road-CAL.txt --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist destination

echo "web-Google"
./sssp --input ./Graphs/web-Google.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist input
echo "***********Source***************"
./sssp --input ./Graphs/web-Google.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist source
echo "***********Destination***************"
./sssp --input ./Graphs/web-Google.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist destination

echo "amazon0312"
./sssp --input ./Graphs/amazon0312.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist input
echo "***********Source***************"
./sssp --input ./Graphs/amazon0312.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist source
echo "***********Destination***************"
./sssp --input ./Graphs/amazon0312.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync outcore --edgelist destination

echo "Configuration $1,$2 : Work-effecient Parallel Method No Shared Memory Incore"

echo "roadNet-CA Graph Results"
./sssp --input ./Graphs/roadNet-CA.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist input
echo "***********Source***************"
./sssp --input ./Graphs/roadNet-CA.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist source
echo "***********Destination***************"
./sssp --input ./Graphs/roadNet-CA.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist destination


echo "msdoor"
./sssp --input ./Graphs/msdoor.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist input
echo "***********Source***************"
./sssp --input ./Graphs/msdoor.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist source
echo "***********Destination***************"
./sssp --input ./Graphs/msdoor.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist destination

echo "road-CAL.txt"
./sssp --input ./Graphs/road-CAL.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist input
echo "***********Source***************"
./sssp --input ./Graphs/road-CAL.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist source
echo "***********Destination***************"
./sssp --input ./Graphs/road-CAL.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist destination

echo "web-Google"
./sssp --input ./Graphs/web-Google.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist input
echo "***********Source***************"
./sssp --input ./Graphs/web-Google.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist source
echo "***********Destination***************"
./sssp --input ./Graphs/web-Google.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist destination

echo "amazon0312"
./sssp --input ./Graphs/amazon0312.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist input
echo "***********Source***************"
./sssp --input ./Graphs/amazon0312.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist source
echo "***********Destination***************"
./sssp --input ./Graphs/amazon0312.txt  --bsize $1 --bcount $2 --output output.txt --method tpe --usesmem no --sync incore --edgelist destination

