---
marp: true
style: |
  section h1 {
    text-align: center;
    }
  .column {
    float: left;
    width: 50%;
    outline: 20px solid #FFF;
    border: 20px solid #AAF;
    background-color: #AAF;
    }
  .row:after {
    display: table;
    clear: both;
    }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  .columns-left {
    background: none;
  }
  .columns-right {
    background: none;
  }
---

<!-- paginate: true -->

# Welcome to the

# Introduction to GPUs course

Stockholm/Zoom, September - October, 2023

### A practical course presented by PDC in collaboration with ENCCS, HPE and AMD

#
### PDC Center for High Performance Computing, KTH Royal Institute of Technology, Sweden

---

# Course curriculum

### GPU computing concepts and programming model

### GPU programming models: HIP, SYCL, OpenMP

### Compiler techniques

### Performance tools


---

# Course schedule

### The course is held
* 15 & 22 September (afternoons only, online)
* 12 & 13 October (all day, in person at KTH, Stockholm)
* Time in between the sessions for own work on exercises

---

### Resource and important links

* Schedule: https://github.com/PDC-support/introduction-to-gpu
* Course material: https://github.com/PDC-support/introduction-to-gpu
* HackMD: https://hackmd.io/@johanhellsvik/IntroductionToGPU
* Slack: link shared over email

### The HackMD document and the Slack board will be open during all the course.

* Questions and answers in the HackMD will be curated so that we can keep the document for later.
* The Slack is transient (90 days, free version of Slack), wherefore we aim to migrate contents to HackMD whenever it is relevant.
