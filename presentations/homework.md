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

# Introduction to GPUs course

Stockholm/Zoom, September - October, 2023

### A practical course presented by PDC in collaboration with ENCCS, HPE and AMD

#
#
#
### PDC Center for High Performance Computing, KTH Royal Institute of Technology, Sweden

---

# Homework

The two onsite days are three weeks from now. We encourage you to

* work through the remaining shorter exercises for HIP, SYCL, OpenMP,
* think about how you would like to implement Himeno in chosen framework at onsite days,
* optional: think about how you would like to implement your own algorithm in chosen framework.

### Questions and answers can be posted on the course HackMD page, and on the Slack board.



