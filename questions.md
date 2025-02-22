# questions for organizers 

- what is exact form of the input? is it latituede and longitude? or is it a set of lats and longs 

# what is the difference between trip volume and trip sample 

why is the sample sometimes bigger than the trip volume 

# if it is what i think it is, where the sample is just there for confidence 

- then where is the time dimension, that volume of trips happened when? can we assume it is static for all columns?

- are teh workshops recorded anywhere?


# question: 
- we have the id of different segments 
- based on the segment can we get nearest neighbors 
  - Euclidian distance 
    - simple
    - kd tree
  - then connecting
    - build graph database for this idea
    - but there are parallel segments that never connect 
  - some hybrid score based on connecting or euclidian
  - incorporate direction 
    - how do we incorporate that field for a segment? 
  - any other fields we need to take into account for the score?