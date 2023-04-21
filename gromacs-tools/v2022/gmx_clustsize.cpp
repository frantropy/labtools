/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 1991- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

using clustsize_t = float;
#include "gmxpre.h"

#include <cmath>
#include <execution>
#include <algorithm>
#include <functional> 
#include <numeric>

#include "gromacs/commandline/filenm.h"
#include "gromacs/commandline/pargs.h"
#include "gromacs/fileio/matio.h"
#include "gromacs/fileio/tpxio.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/gmxana/gstat.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/mtop_lookup.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/smalloc.h"

static void clust_size(const char*             ndx,
                       const char*             trx,
                       const char*             xpm,
                       const char*             xpmw,
                       const char*             ncl,
                       const char*             acl,
                       const char*             mcl,
                       const char*             histo,
                       const char*             histotime,
                       const char*             clustime,
                       const char*             trmatrix,
                       const char*             kmatrix,
                       const char*             tempf,
                       const char*             mcn,
                       gmx_bool                bMol,
                       gmx_bool                bPBC,
                       const char*             tpr,
                       clustsize_t                  cut,
                    //    clustsize_t                  cutoffino,
                       clustsize_t                  mol_cut,
                       int                     bOndx,
                       int                     nskip,
                       int                     skip_last_nmol,
                       int                     nlevels,
                       t_rgb                   rmid,
                       t_rgb                   rhi,
                       int                     ndf,
                       const gmx_output_env_t* oenv)
{
    FILE *       fp, *gp, *hp, *tp, *cndx;
    int*         index = nullptr;
    int          nindex, natoms;
    t_trxstatus* status;
    rvec *       x = nullptr, *v = nullptr, *xcm = nullptr, dx;
    t_pbc        pbc;
    gmx_bool     bSame, bTPRwarn = TRUE;
    /* Topology stuff */
    t_trxframe    fr;
    TpxFileHeader tpxh;
    gmx_mtop_t    mtop;
    PbcType       pbcType = PbcType::Unset;
    int           ii, jj;
    double        temp, tfac;
    /* Cluster size distribution (matrix) */
    double** cs_dist = nullptr;
    double** tr_matrix = nullptr;
    double** rate_matrix = nullptr;
    double* norm_matrix = nullptr;
    bool* norm_done = nullptr;
    double   tf, dx2, cut2, mcut2, *t_x = nullptr, *t_y, cmid, cmax, cav, ekin;
    int    i, j, k, ai, aj, ci, cj, nframe, nclust, n_x, max_size = 0;
    int *  clust_index, *index_size, *index_old_size, *clust_size, *clust_written, max_clust_size, max_clust_ind, nav, nhisto;
    t_rgb  rlo          = { 1.0, 1.0, 1.0 };
    int    frameCounter = 0;
    double frameTime;

    clear_trxframe(&fr, TRUE);
    auto timeLabel = output_env_get_time_label(oenv);
    tf             = output_env_get_time_factor(oenv);
    fp             = xvgropen(ncl, "Number of clusters", timeLabel, "N", oenv);
    gp             = xvgropen(acl, "Average cluster size", timeLabel, "#molecules", oenv);
    hp             = xvgropen(mcl, "Max cluster size", timeLabel, "#molecules", oenv);
    tp             = xvgropen(tempf, "Temperature of largest cluster", timeLabel, "T (K)", oenv);

    if (!read_first_frame(oenv, &status, trx, &fr, TRX_NEED_X | TRX_READ_V))
    {
        gmx_file(trx);
    }

    natoms = fr.natoms;
    x      = fr.x;

    if (tpr)
    {
        tpxh = readTpxHeader(tpr, true);
        if (tpxh.natoms != natoms)
        {
            gmx_fatal(FARGS, "tpr (%d atoms) and trajectory (%d atoms) do not match!", tpxh.natoms, natoms);
        }
        pbcType = read_tpx(tpr, nullptr, nullptr, &natoms, nullptr, nullptr, &mtop);
    }
    if (ndf <= -1)
    {
        tfac = 1;
    }
    else
    {
        tfac = ndf / (3.0 * natoms);
    }

    gmx::RangePartitioning mols;
    if (bMol)
    {
        if (ndx)
        {
            printf("Using molecules rather than atoms. Not reading index file %s\n", ndx);
        }
        GMX_RELEASE_ASSERT(tpr, "Cannot access topology without having read it from TPR");
        mols = gmx_mtop_molecules(mtop);

        /* Make dummy index */
        nindex = mols.numBlocks()-skip_last_nmol;
        snew(index, nindex);
        for (i = 0; (i < nindex); i++)
        {
            index[i] = i;
        }
    }
    else
    {
        char* gname;
        rd_index(ndx, 1, &nindex, &index, &gname);
        sfree(gname);
    }

    snew(clust_index, nindex);
    snew(index_size, nindex);
    snew(index_old_size, nindex);
    snew(clust_size, nindex);
    snew(xcm, nindex);
    /* transition matrix */
    snew(tr_matrix, nindex);
    for(i=0;i<nindex;i++) snew(tr_matrix[i], nindex);
    /* rate matrix */
    snew(rate_matrix, nindex);
    for(i=0;i<nindex;i++) snew(rate_matrix[i], nindex);
    /* norm needed to calculate the rate and transition matrices */
    snew(norm_matrix, nindex);
    /* flag to accumulate correctly the norm matrix */
    snew(norm_done, nindex);
    mcut2 = mol_cut*mol_cut;
    cut2   = cut * cut;
    // total number of trajectory frames
    nframe = 0;
    // number of analysed frames
    n_x    = 0;
    snew(t_y, nindex);
    for (i = 0; (i < nindex); i++)
    {
        t_y[i] = i + 1;
    }
    max_clust_size = 1;
    max_clust_ind  = -1;
    int molb       = 0;
    cndx = xvgropen(clustime, "Index of the oligomer to which each monomer belongs", timeLabel, "Monomer index", oenv);
    double frameTimeStep=1.;
    do
    {
        if(nframe==1&&fr.bTime) frameTimeStep=fr.time;
        if ((nskip == 0) || ((nskip > 0) && ((nframe % nskip) == 0)))
        {
            if (bPBC)
            {
                set_pbc(&pbc, pbcType, fr.box);
            }
            max_clust_size = 1;
            max_clust_ind  = -1;

            /* Put all atoms/molecules in their own cluster, with size 1 */
            for (i = 0; (i < nindex); i++)
            {
                /* Cluster index is indexed with atom index number */
                clust_index[i] = i;
                /* Cluster size is indexed with cluster number */
                clust_size[i] = 1;
                /* Initially each molecule belongs to a cluster of size 1 */
                index_size[i] = 1;
                /* Flag to accumulate the norm matrix */
                norm_done[i] = FALSE;
            }
            /* calculate the center of each molecule */
            for (i = 0; (i < nindex); i++)
            {   
                clear_rvec(xcm[i]);
                ai = index[i];
                double tm = 0.;
                for (ii = mols.block(ai).begin(); ii < mols.block(ai).end(); ii++)
                {
                    for (int m = 0; (m < DIM); m++)
                    {
                        xcm[i][m] += x[ii][m];
                    }
                    tm += 1.0; 
                }
                for (int m = 0; (m < DIM); m++)
                {
                    xcm[i][m] /= tm;
                }
            }

            /* Loop over atoms/molecules */
            for (i = 0; (i < nindex); i++)
            {
                ai = index[i];
                ci = clust_index[i];

                /* Loop over atoms/molecules (only half a matrix) */
                for (j = i + 1; (j < nindex); j++)
                {
                    cj = clust_index[j];

                    if (bPBC)
                    {
                        pbc_dx(&pbc, xcm[i], xcm[j], dx);
                    }
                    else
                    {
                        rvec_sub(xcm[i], xcm[j], dx);
                    }
                    dx2   = iprod(dx, dx);

                    if (dx2 > mcut2) continue;

                    /* If they are not in the same cluster already */
                    if (ci != cj)
                    {
                        aj = index[j];

                        /* Compute distance */
                        if (bMol)
                        {
                            GMX_RELEASE_ASSERT(mols.numBlocks() > 0,
                                               "Cannot access index[] from empty mols");
                            bSame = FALSE;
                            for (ii = mols.block(ai).begin(); !bSame && ii < mols.block(ai).end(); ii++)
                            {
                                for (jj = mols.block(aj).begin(); !bSame && jj < mols.block(aj).end(); jj++)
                                {
                                    if (bPBC)
                                    {
                                        pbc_dx(&pbc, x[ii], x[jj], dx);
                                    }
                                    else
                                    {
                                        rvec_sub(x[ii], x[jj], dx);
                                    }
                                    dx2   = iprod(dx, dx);
                                    bSame = (dx2 < cut2);
                                }
                            }
                        }
                        else
                        {
                            if (bPBC)
                            {
                                pbc_dx(&pbc, x[ai], x[aj], dx);
                            }
                            else
                            {
                                rvec_sub(x[ai], x[aj], dx);
                            }
                            dx2   = iprod(dx, dx);
                            bSame = (dx2 < cut2);
                        }
                        /* If distance less than cut-off */
                        if (bSame)
                        {
                            /* Merge clusters: check for all atoms whether they are in
                             * cluster cj and if so, put them in ci
                             */
                            for (k = 0; (k < nindex); k++)
                            {
                                if (clust_index[k] == cj)
                                {
                                    if (clust_size[cj] <= 0)
                                    {
                                        gmx_fatal(FARGS,
                                                  "negative cluster size %d for element %d",
                                                  clust_size[cj],
                                                  cj);
                                    }
                                    clust_size[cj]--;
                                    clust_index[k] = ci;
                                    clust_size[ci]++;
                                }
                            }
                        }
                    }
                }
            }
            for (k = 0; (k < nindex); k++)
            {
                 // this tells how large is the cluster to which each molecule belongs
                 index_size[k] = clust_size[clust_index[k]];
            }
            n_x++;
            srenew(t_x, n_x);
            if (fr.bTime)
            {
                frameTime = fr.time;
            }
            else if (fr.bStep)
            {
                frameTime = fr.step;
            }
            else
            {
                frameTime = ++frameCounter;
            }
            t_x[n_x - 1] = frameTime * tf;
            srenew(cs_dist, n_x);
            snew(cs_dist[n_x - 1], nindex);
            nclust = 0;
            cav    = 0;
            nav    = 0;
            for (i = 0; (i < nindex); i++)
            {
                ci = clust_size[i];
                if (ci > max_clust_size)
                {
                    max_clust_size = ci;
                    max_clust_ind  = i;
                }
                if (ci > 0)
                {
                    nclust++;
                    /* this is the cluster size time-resolved distribution 
                       that is cs[frame][i]=# of oligomers of order (i+1) */
                    cs_dist[n_x - 1][ci - 1] += 1.0;
                    max_size = std::max(max_size, ci);
                    if (ci > 1)
                    {
                        cav += ci;
                        nav++;
                    }
                }
            }
            fprintf(fp, "%14.6e  %10d\n", frameTime, nclust);
            if (nav > 0)
            {
                fprintf(gp, "%14.6e  %10.3f\n", frameTime, cav / nav);
            }
            fprintf(hp, "%14.6e  %10d\n", frameTime, max_clust_size);
            /* update the transition matrix */
            if (n_x>1) 
            {
                double Volume = det(fr.box)*0.0006022;  // NA * nm3->m3
                double Volume2 = Volume*Volume;
                for(i=0;i<nindex;i++)
                {
                   // transition from an oligomer of order index_old_size[i] to on of order index_size[i] 
                   if(cs_dist[n_x-2][index_old_size[i]-1]>0.)
                   {
                     tr_matrix[index_size[i]-1][index_old_size[i]-1]+=1./(cs_dist[n_x-2][index_old_size[i]-1]*((double)index_old_size[i]));
                     if(index_old_size[i]>index_size[i]) {
                       /* k_off */
                       /* this is 1/([oligomer]) that are dissociating */
                       rate_matrix[index_size[i]-1][index_old_size[i]-1]+=Volume/(cs_dist[n_x-2][index_old_size[i]-1]*((double)index_old_size[i]));
                     } else if(index_old_size[i]<index_size[i]){
                       /* k_on */
                       /* this is 1/([oligomer_ligand][oligomer_reactants]) that are associating */
                       double fact=0;
                       for(j=0;j<(index_size[i]-index_old_size[i]);j++) {
                         fact+=cs_dist[n_x-2][j]*((double)(j+1));
                       }
                       fact -= index_old_size[i];
                       rate_matrix[index_size[i]-1][index_old_size[i]-1]+=Volume2/(cs_dist[n_x-2][index_old_size[i]-1]*((double)index_old_size[i])*fact);
                     }
 
                     if(!norm_done[index_old_size[i]-1]) norm_matrix[index_old_size[i]-1]+=1.0;
                     norm_done[index_old_size[i]-1] = TRUE;
                   }
                }
            }
            /* save index_size so that it can be used to generate the transition matrix */
            for(i=0;i<nindex;i++) index_old_size[i] = index_size[i]; 
        }
        /* Analyse velocities, if present */
        if (fr.bV)
        {
            if (!tpr)
            {
                if (bTPRwarn)
                {
                    printf("You need a [REF].tpr[ref] file to analyse temperatures\n");
                    bTPRwarn = FALSE;
                }
            }
            else
            {
                v = fr.v;
                /* Loop over clusters and for each cluster compute 1/2 m v^2 */
                if (max_clust_ind >= 0)
                {
                    ekin = 0;
                    for (i = 0; (i < nindex); i++)
                    {
                        if (clust_index[i] == max_clust_ind)
                        {
                            ai     = index[i];
                            double m = mtopGetAtomMass(mtop, ai, &molb);
                            ekin += 0.5 * m * iprod(v[ai], v[ai]);
                        }
                    }
                    temp = (ekin * 2.0) / (3.0 * tfac * max_clust_size * gmx::c_boltz);
                    fprintf(tp, "%10.3f  %10.3f\n", frameTime, temp);
                }
            }
        }
        fprintf(cndx, "%10.3f ", frameTime);
        for (i = 0; (i < nindex); i++) fprintf(cndx, "%i ", clust_index[i]);
        fprintf(cndx, "\n");

        if((bOndx>1) && (bMol)) 
        {
            int largest=bOndx;
            /* index file per frame per size */
            snew(clust_written, nindex);
            for(int oligsize=2;oligsize<=largest;oligsize++) {
                std::string ndx_name = "cs_" +std::to_string(oligsize) + "_" + std::to_string(nframe) + ".ndx";
                fp = gmx_ffopen(ndx_name.c_str(), "w");
                for (int i = 0; (i < nindex); i++)
                {
                    // this tells how large is the cluster to which each molecule belongs
	            ci = clust_index[i];
	            if(clust_written[ci]==1) continue;
                    if(index_size[i] == oligsize) 
                    {
                        fprintf(fp, "[ clust %i ]\n", ci);
                        for (int j : mols.block(i)) fprintf(fp, "%d\n", j+1);
	                for(int j=i+1; (j < nindex); j++)
	                {
	                    if(clust_index[j]==ci) 
	                    {
                                for (int k : mols.block(j)) fprintf(fp, "%d\n", k+1);
                            }
	                }
	                clust_written[ci]=1;
	            }
	        }
                gmx_ffclose(fp);
            }
            sfree(clust_written);
        }

        nframe++;
    } while (read_next_frame(oenv, status, &fr));
    close_trx(status);
    done_frame(&fr);
    xvgrclose(fp);
    xvgrclose(gp);
    xvgrclose(hp);
    xvgrclose(tp);
    xvgrclose(cndx); 

    snew(clust_written, nindex);
    if (max_clust_ind >= 0)
    {
        fp = gmx_ffopen(mcn, "w");
      /* CARLO: this adds the indices for all the clusters at the end of the trajectory */
      if (bMol)
      {
        for (int i = 0; (i < nindex); i++)
        {
	  ci = clust_index[i];
	  if(clust_written[ci]==1) continue;
          fprintf(fp, "[ clust %i ]\n", ci);
          for (int j : mols.block(i))
          {
             fprintf(fp, "%d\n", j+1);
          }
	  for(int j=i+1; (j < nindex); j++)
	  {
	    if(clust_index[j]==ci) 
	    {
              for (int k : mols.block(j))
              {
                fprintf(fp, "%d\n", k+1);
              }
	    }
	  }
	  clust_written[ci]=1;
	}
      }
        fprintf(fp, "[ max_clust ]\n");
        for (i = 0; (i < nindex); i++)
        {
            if (clust_index[i] == max_clust_ind)
            {
                if (bMol)
                {
                    GMX_RELEASE_ASSERT(mols.numBlocks() > 0,
                                       "Cannot access index[] from empty mols");
                    for (int j : mols.block(i))
                    {
                        fprintf(fp, "%d\n", j + 1);
                    }
                }
                else
                {
                    fprintf(fp, "%d\n", index[i] + 1);
                }
            }
        }
        gmx_ffclose(fp);
    }

    /* Print the real distribution cluster-size/numer, averaged over the trajectory. */
    fp     = xvgropen(histo, "Cluster size distribution", "Cluster size", "()", oenv);
    nhisto = 0;
    fprintf(fp, "%5d  %8.3f\n", 0, 0.0);
    for (j = 0; (j < max_size); j++)
    {
        double nelem = 0;
        for (i = 0; (i < n_x); i++)
        {
            nelem += cs_dist[i][j];
        }
        fprintf(fp, "%5d  %8.3f\n", j + 1, nelem / n_x);
        nhisto += static_cast<int>((j + 1) * nelem / n_x);
    }
    fprintf(fp, "%5d  %8.3f\n", j + 1, 0.0);
    xvgrclose(fp);

    fp = xvgropen(histotime, "Time Resolved distribution of oligomers order", timeLabel, "# of oligomers of order #", oenv);
    for (i = 0; (i < n_x); i++)
    {
        fprintf(fp, "%14.6e ", t_x[i]);
    	for (j = 0; (j < max_size); j++)
    	{
        	fprintf(fp, " %8.3f", cs_dist[i][j]);
        }
        fprintf(fp,"\n");
    }
    xvgrclose(fp); 

    fp = xvgropen(trmatrix, "Transition Matrix", "Oligomers order", "Oligomers order", oenv);
    /* The sum of the rows should be divisible for the oligomer order (that is the row number)\n");
       Rows are transitions toward lower order oligomers\n");
       Columns are transitions toward higher order oligomers\n"); */
    for (i = 0; (i < nindex); i++)
    {
    	for (j = 0; (j < nindex); j++)
    	{
        	fprintf(fp, "%8.6lf ", tr_matrix[i][j]/norm_matrix[j]);
        }
        fprintf(fp,"\n");
    }
    xvgrclose(fp);
 
    fp = xvgropen(kmatrix, "Rates Matrix in ps-1", "Oligomers order", "Oligomers order", oenv);
    /* Rows are transitions toward lower order oligomers\n");
       Columns are transitions toward higher order oligomers\n"); */
    for (i = 0; (i < nindex); i++)
    {
    	for (j = 0; (j < nindex); j++)
    	{
        	fprintf(fp, "%8.6lf ", rate_matrix[i][j]/norm_matrix[j]/frameTimeStep);
        }
        fprintf(fp,"\n");
    }
    xvgrclose(fp);

    fprintf(stderr, "Total number of atoms in clusters =  %d\n", nhisto);

    /* Look for the smallest entry that is not zero
     * This will make that zero is white, and not zero is coloured.
     */
    cmid = 100.0;
    cmax = 0.0;
    for (i = 0; (i < n_x); i++)
    {
        for (j = 0; (j < max_size); j++)
        {
            if ((cs_dist[i][j] > 0) && (cs_dist[i][j] < cmid))
            {
                cmid = cs_dist[i][j];
            }
            cmax = std::max(cs_dist[i][j], cmax);
        }
    }
    fprintf(stderr, "cmid: %g, cmax: %g, max_size: %d\n", cmid, cmax, max_size);
    cmid = 1;
    fp   = gmx_ffopen(xpm, "w");
    /*write_xpm3(fp,
               0,
               "Cluster size distribution",
               "# clusters",
               timeLabel,
               "Size",
               n_x,
               max_size,
               t_x,
               t_y,
               cs_dist,
               0,
               cmid,
               cmax,
               rlo,
               rmid,
               rhi,
               &nlevels);*/
    gmx_ffclose(fp);
    cmid = 100.0;
    cmax = 0.0;
    for (i = 0; (i < n_x); i++)
    {
        for (j = 0; (j < max_size); j++)
        {
            cs_dist[i][j] *= (j + 1);
            if ((cs_dist[i][j] > 0) && (cs_dist[i][j] < cmid))
            {
                cmid = cs_dist[i][j];
            }
            cmax = std::max(cs_dist[i][j], cmax);
        }
    }
    fprintf(stderr, "cmid: %g, cmax: %g, max_size: %d\n", cmid, cmax, max_size);
    fp = gmx_ffopen(xpmw, "w");
    /*write_xpm3(fp,
               0,
               "Weighted cluster size distribution",
               "Fraction",
               timeLabel,
               "Size",
               n_x,
               max_size,
               t_x,
               t_y,
               cs_dist,
               0,
               cmid,
               cmax,
               rlo,
               rmid,
               rhi,
               &nlevels);*/
    gmx_ffclose(fp);
    sfree(t_x);
    sfree(t_y);
    for (i = 0; (i < n_x); i++)
    {
        sfree(cs_dist[i]);
    }
    sfree(cs_dist);
    sfree(clust_index);
    sfree(clust_size);
    sfree(index);
}

static inline void kernel_density_estimator(std::vector<clustsize_t> &x, const std::vector<clustsize_t> &bins, const clustsize_t mu, const clustsize_t norm) {
    clustsize_t h = 0.01;
    clustsize_t from_x = std::max(mu - 2 * h, bins[0]);
    clustsize_t to_x = std::min(mu + 2 * h, bins.back());
    auto is_geq_start = [&from_x](clustsize_t i){ return i >= from_x; };
    auto is_geq_end = [&to_x](clustsize_t i){ return i > to_x; };
    auto start = std::find_if(bins.begin(), bins.end(), is_geq_start);
    auto end = std::find_if(bins.begin(), bins.end(), is_geq_end);
    int from = std::distance(bins.begin(), start);
    int to = std::distance(bins.begin(), end);
    clustsize_t scale = norm/(0.73853587f * h * std::sqrt(2.f*M_PIf));
    if(mu<h) scale *= 2.;
    clustsize_t shift = std::exp(-2.f);
    for (int i = from; i < to; i++) {
        clustsize_t f=(mu-bins[i])/h;
        clustsize_t kernel = std::exp(-0.5f*f*f);
        x[i] += scale*(kernel-shift);
    }
}

static inline clustsize_t calc_mean(const std::vector<clustsize_t> &v, const clustsize_t dx)
{
    clustsize_t dm = 0.;
    clustsize_t norm = 0.;
    for(auto it = v.begin(); it != v.end(); ++it) {
       unsigned i = std::distance(v.begin(), it);
       if(v[i]>0.) {
          clustsize_t d = (dx*static_cast<clustsize_t>(i)+0.5*dx);
          dm+=v[i]*d;
          norm+=v[i];
       }
    }
    if (norm==0.) norm = 1.;
    return dm/norm;
}

static inline clustsize_t calc_prob(const std::vector<clustsize_t> &v, const clustsize_t dx)
{
    clustsize_t prob = 0.;
    for(auto it = v.begin(); it != v.end(); ++it) {
       unsigned i = std::distance(v.begin(), it);
       if(v[i]>0.) {
          prob += v[i]*dx;
       }
    }
    if(prob>1.f) prob = 1.f;
    return prob;
}


static inline int is_flag_tolerance(const std::vector<clustsize_t> &v)
{
    std::vector<clustsize_t> rv(v.size());
    std::reverse_copy(v.begin(), v.end(), rv.begin());

    // keep track of the minimum value
    clustsize_t min=rv[0];
    int danger=0;

    for ( int i = 0; i < rv.size()-1; ++i )
    {
        if(rv[i+1]<min) min=rv[i+1];
        if((rv[i] - rv[i+1])<0.)
        {
           // in the beginning we accept at most one false point and only for small differences
           if(i==0) {
             danger=1;
             if(((min-rv[i+1]))<-10e-05) return 2;
           } 
           if(danger&&i==1) return 2;
           // -10e3 could also be fine (to make the check weaker if needed)
           // -10e5 could also be fine (to make the check stronger if needed)
           if(((min-rv[i+1]))<-10e-02) return 2;
        }
    }
    return 0;
}

static inline int is_slope(const std::vector<clustsize_t> &v, const clustsize_t dx, const clustsize_t prob)
{
    std::vector<clustsize_t> slope(v.size()-2);
    unsigned counter = 0;
    clustsize_t min_d = 0;
    for ( auto it = v.begin()+1; it < v.end()-1; ++it )
    {
        unsigned i = std::distance(v.begin(), it);
        slope[counter] = (v[i+1] - v[i-1]) / (2.*dx);
        if(!min_d&&v[i]>0.) min_d=(dx*static_cast<clustsize_t>(i)+0.5*dx);
        counter++;
    }

    clustsize_t means=0., c=1., max_means=0., int_min_means=0.;
    int danger=0;
    for (int i=0; i < slope.size(); i++)
    {
        if(slope[i]!=0.) {
           means += (slope[i]/prob-means)/c;
           // if the means drop too much we are in danger; is not a single gaussian
           if(max_means>10. && max_means>means) {
              if(!danger) int_min_means = means;
              else {
                // if the mean continues to drop just go on checking it
                if(int_min_means>=means) int_min_means=means;
                // if the mean start again to increase GOT IT!
                else return 1;
              } 
              danger=1;
           }
           if(means>max_means) max_means = means;
           c++;
        }
    }
    // range 90-130, weaker-stronger
    // this number also depend from the sigma
    //if(means<(65.*(1.+min_d))) return 1;
    if(means<(90.*(1.+(min_d-0.3)))) return 1;

    return 0;
}

static inline clustsize_t calc_sigma(const std::vector<clustsize_t> &v, const clustsize_t dx)
{
    clustsize_t n_bins = static_cast<clustsize_t>(v.size());
    std::vector<clustsize_t> rv(v.size());
    std::reverse_copy(v.begin(), v.end(), rv.begin());
    auto j3 = std::adjacent_find(rv.begin(), rv.end(), [](clustsize_t a, clustsize_t b) {return (a<=b) && (b!=0);});
    clustsize_t dm = 0.;
    clustsize_t d2m = 0.;
    clustsize_t norm = 0.;
    if(j3!=rv.end()) {
       for(auto it = v.begin()+1; it != v.end()-std::distance(rv.begin(), j3); ++it) {
          unsigned i = std::distance(v.begin(), it);
          if(v[i]>0.) {
             clustsize_t d = (dx*static_cast<clustsize_t>(i)+0.5*dx);
             dm+=v[i] * d;
             d2m+=v[i] * d * d;
             norm += v[i];
          }
       }
    } else {
       return -1;
    }
    if (norm == 0.) norm = 1.;
    clustsize_t sigma = std::sqrt(d2m/norm - dm*dm/norm/norm);
    return sigma;
}

static inline double is_dist(const std::vector<clustsize_t> &x, const double dx, const double sigma, const int flag)
{
    std::vector<double> v(x.size());
    std::vector<double> rv(v.size());
    auto to_double = [](const clustsize_t i) { return static_cast<double>(i); };
    std::transform(std::cbegin(x), std::cend(x), std::begin(v), to_double);
    // std::vector<double> rv(v.size());
    std::reverse_copy(v.begin(), v.end(), rv.begin());
    auto j3 = std::adjacent_find(rv.begin(), rv.end(), [](double a, double b) {return (a<=b) && (b!=0);});
    auto first_val = find_if( std::begin(v), std::end(v), [](auto x) { return x != 0; });
    auto last_val = find_if( std::rbegin(v), std::rend(v), [](auto x) { return x != 0; });
    auto until = v.end()-std::distance(rv.begin(), j3);
    double mind = (dx*static_cast<double>(std::distance(v.begin(),first_val))+0.5*dx);
    double maxd = std::min((dx*static_cast<double>(std::distance(v.begin(),(last_val+1).base()))+0.5*dx),(dx*static_cast<double>(std::distance(v.begin(),until))+0.5*dx));
    double sigma_cut = (maxd-mind)/8.; 
    if (sigma > sigma_cut) until = until - std::distance(first_val, until)/2;

    if(flag==2) {
       double d12 = 0.;
       double norm = 0.;
       for(auto it = v.begin()+1; it != until; ++it) { 
          unsigned i = std::distance(v.begin(), it);
          if(v[i]>0.) {
             double d = (dx*static_cast<double>(i)+0.5*dx);
             d12+=v[i]*std::pow(1./d,12.);
             norm+=v[i];
          }
       }
       if (norm == 0.) norm = 1.;
       d12 = (d12>0. ? std::pow(d12/norm, -1./12.):0.);
       return d12;
    } else {
      double dexp = 0.;
      double norm = 0.;
      for(auto it = v.begin()+1; it != v.end(); ++it) { 
         unsigned i = std::distance(v.begin(), it);
         if(v[i]>0.) {
            double d = (dx*static_cast<double>(i)+0.5*dx);
            dexp+=v[i]*std::exp(1./d/0.05);
            norm+=v[i];
         }
      }
      if (norm == 0.) norm = 1.;
      dexp = (dexp>0. ? (1./0.05)/std::log(dexp/norm):0.);
      return dexp;
    }
}



#define NBINS 4
static inline int n_bins(const clustsize_t cut, const clustsize_t factor = 1.0)
{
    return cut / (0.01 / factor);
}

static void do_interm_mat(const char*             trx,
                          const char*             outfile_inter,
                          const char*             outfile_intra,
                          gmx_bool                bPBC,
                          const char*             tpr,
                          clustsize_t                  cut,
                          clustsize_t                  mol_cut,
                          clustsize_t                  d_pow,
                          int                     nskip,
                          int                     skip_last_nmol,
                          gmx_bool                write_histo,
                          const gmx_output_env_t* oenv)
{
    t_trxframe    fr;
    clear_trxframe(&fr, TRUE);

    t_trxstatus* status;
    if (!read_first_frame(oenv, &status, trx, &fr, TRX_NEED_X | TRX_READ_V))
    {
        gmx_file(trx);
    }

    int natoms = fr.natoms;
    rvec *x = fr.x;

    TpxFileHeader tpxh;
    gmx_mtop_t    mtop;
    PbcType pbcType = PbcType::Unset;
    if (tpr)
    {
        tpxh = readTpxHeader(tpr, true);
        if (tpxh.natoms != natoms)
        {
            gmx_fatal(FARGS, "tpr (%d atoms) and trajectory (%d atoms) do not match!", tpxh.natoms, natoms);
        }
        pbcType = read_tpx(tpr, nullptr, nullptr, &natoms, nullptr, nullptr, &mtop);
    }

    gmx::RangePartitioning mols;
    GMX_RELEASE_ASSERT(tpr, "Cannot access topology without having read it from TPR");
    mols = gmx_mtop_molecules(mtop);

    // number of molecules
    int nindex = mols.numBlocks()-skip_last_nmol;
    std::vector<int> num_mol;
    num_mol.push_back(1);
    int num_unique_molecules=0;
    // number of atoms per molecule, assuming them identical when consecutive molecules have the same number of atoms
    std::vector<int> natmol2;
    natmol2.push_back(mols.block(0).end());
    for(int i=1; i<nindex; i++) {
       natmol2.push_back(mols.block(i).end()-mols.block(i-1).end());
       if(natmol2[i]==natmol2[i-1]) num_mol[num_unique_molecules]++;
       else {
         num_mol.push_back(1);
         num_unique_molecules++;
       }
    }
    std::vector<int>::iterator it = std::unique(natmol2.begin(), natmol2.end());  
    natmol2.resize(std::distance(natmol2.begin(),it));

    std::vector<int> start_index;
    std::vector<int> mol_id;
    mol_id.push_back(0);
    std::vector<clustsize_t> inv_num_mol;
    start_index.push_back(0); 
    num_unique_molecules=0;
    inv_num_mol.push_back(1./(static_cast<clustsize_t>(num_mol[num_unique_molecules])));

    for(int i=1; i<nindex; i++) {
       if(mols.block(i).end()-mols.block(i-1).end()==natmol2[num_unique_molecules]) {
          start_index.push_back(start_index[i-1]);
       } else {
          start_index.push_back(natmol2[num_unique_molecules]);
          num_unique_molecules++;
       }
       mol_id.push_back(num_unique_molecules);
       inv_num_mol.push_back(1./static_cast<clustsize_t>(num_mol[num_unique_molecules]));
    }

    printf("number of different molecules %lu\n", natmol2.size());
    for(std::size_t i=0; i<natmol2.size();i++) printf("mol %lu num %u size %u\n", i, num_mol[i], natmol2[i]);
    //for(unsigned i=0; i<nindex;i++) printf("start_idnex %u %u\n", i, start_index[i]);
    //for(unsigned i=0; i<nindex;i++) printf("invnummol %u %lf\n", i, inv_num_mol[i]);

    // Tensor atm x atm x nbins to accumulate density function
    std::vector<std::vector<std::vector<std::vector<clustsize_t> > > > interm_same_mat_density(natmol2.size());
    std::vector<std::vector<std::vector<std::vector<clustsize_t> > > > interm_cross_mat_density((natmol2.size()*(natmol2.size()-1))/2);
    std::vector<std::vector<std::vector<std::vector<clustsize_t> > > > intram_mat_density(natmol2.size());

    std::vector<clustsize_t> density_bins(n_bins(cut, NBINS));
    for (int i = 0; i < density_bins.size(); i++ ) density_bins[i] = cut/static_cast<clustsize_t>(density_bins.size())*static_cast<clustsize_t>(i)+cut/static_cast<clustsize_t>(density_bins.size()*2);

    int cross_count=0;
    std::vector<std::vector<int> > cross_index(natmol2.size(), std::vector<int>(natmol2.size(),0));
    for(std::size_t i=0; i<natmol2.size();i++) {
      interm_same_mat_density[i].resize(natmol2[i], std::vector<std::vector<clustsize_t>>(natmol2[i], std::vector<clustsize_t>(n_bins(cut, NBINS),0)));
      intram_mat_density[i].resize(natmol2[i], std::vector<std::vector<clustsize_t>>(natmol2[i], std::vector<clustsize_t>(n_bins(cut, NBINS),0)));
      for(std::size_t j=i+1; j<natmol2.size();j++) {
        interm_cross_mat_density[i].resize(natmol2[i], std::vector<std::vector<clustsize_t>>(natmol2[j], std::vector<clustsize_t>(n_bins(cut, NBINS),0)));
        cross_index[i][j]=cross_count;
        cross_count++;
      }
    } 

    // vector of center of masses
    rvec *xcm = nullptr;
    snew(xcm, nindex);
    const char * atomname;

    clustsize_t mcut2 = mol_cut*mol_cut;
    clustsize_t cut_sig_2 = (cut + 0.02) * (cut + 0.02);
    // total number of trajectory frames
    int nframe = 0;
    // number of analysed frames
    int n_x = 0;

    printf("Ready!\n"); fflush(stdout);
    do
    {
        if ((nskip == 0) || ((nskip > 0) && ((nframe % nskip) == 0)))
        {
            t_pbc pbc;
            if (bPBC) set_pbc(&pbc, pbcType, fr.box);

            /* calculate the center of each molecule */
            for (int i = 0; (i < nindex); i++)
            {   
                clear_rvec(xcm[i]);
                clustsize_t tm = 0.;
                for (int ii = mols.block(i).begin(); ii < mols.block(i).end(); ii++)
                {
                    for (int m = 0; (m < DIM); m++)
                    {
                        xcm[i][m] += x[ii][m];
                    }
                    tm += 1.0; 
                }
                for (int m = 0; (m < DIM); m++)
                {
                    xcm[i][m] /= tm;
                }
            }

            /* Loop over molecules */
            for (int i = 0; i < nindex; i++)
            {
                int molbi = i;
                // Temporary structures for intermediate values
                // this is to set that at least on interaction has been found
                // for each molecule we want to count an atom pair no more than once, and we consider the pair with the shorter distance
                // matrices atm x atm for accumulating distances 
                std::vector<std::vector<clustsize_t> > interm_same_mat_mdist(natmol2[mol_id[i]], std::vector<clustsize_t>(natmol2[mol_id[i]], 100.));    
                std::vector<std::vector<clustsize_t> > intram_mat_mdist(natmol2[mol_id[i]], std::vector<clustsize_t>(natmol2[mol_id[i]], 100.));    
                std::vector<std::vector<std::vector<clustsize_t> > > interm_cross_mat_mdist((natmol2.size()*(natmol2.size()-1))/2);
                for (std::size_t j = mol_id[i]+1; j < natmol2.size(); j++) {
                    interm_cross_mat_mdist[cross_index[mol_id[i]][j]].resize(natmol2[mol_id[i]], std::vector<clustsize_t>(natmol2[mol_id[j]], 100.));    
                }
                
                /* Loop over molecules  */
                for (int j = 0; j < nindex; j++)
                {
                    int molbj = j;
                    rvec dx;
                    if(j!=i) {
                      if (bPBC) pbc_dx(&pbc, xcm[i], xcm[j], dx);
                      else rvec_sub(xcm[i], xcm[j], dx);
                      clustsize_t dx2 = iprod(dx, dx);
                      if (dx2 > mcut2) continue;
                    }
                    if(mol_id[i]!=mol_id[j]&&j<i) continue;

                    /* Compute distance */
                    int a_i = 0;
                    GMX_RELEASE_ASSERT(mols.numBlocks() > 0,"Cannot access index[] from empty mols");
                    for (int ii = mols.block(i).begin(); ii < mols.block(i).end(); ii++)
                    {
                        int a_j = 0;
                        // SKIP HYDROGEN
                        mtopGetAtomAndResidueName(mtop, ii, &molbi, &atomname, nullptr, nullptr, nullptr);
                        if (atomname[0]=='H') // WARNING possible ff specificity
                        {
                            a_i++;
                            continue;
                        }
                        for (int jj = mols.block(j).begin(); jj < mols.block(j).end(); jj++)
                        {
                            // SKIP HYDROGEN
                            mtopGetAtomAndResidueName(mtop, jj, &molbj, &atomname, nullptr, nullptr, nullptr);
                            if (atomname[0]=='H') // WARNING possible ff specificity
                            {
                                a_j++;
                                continue;
                            }
                            if (bPBC) pbc_dx(&pbc, x[ii], x[jj], dx);
                            else rvec_sub(x[ii], x[jj], dx);
                            clustsize_t dx2 = iprod(dx, dx);
                            if(dx2 < cut_sig_2) {
                                if(i!=j) { // intermolecular
                                   if(mol_id[i]==mol_id[j]) { // inter same molecule specie
                                      interm_same_mat_mdist[a_i][a_j] = std::min(interm_same_mat_mdist[a_i][a_j], dx2);
                                    //   interm_same_mat_mdist[a_j][a_i] = interm_same_mat_mdist[a_i][a_j];
                                   } else { // inter cross molecule specie
                                      interm_cross_mat_mdist[cross_index[mol_id[i]][mol_id[j]]][a_i][a_j] = std::min(interm_cross_mat_mdist[cross_index[mol_id[i]][mol_id[j]]][a_i][a_j], dx2);
                                   }
                                } else { // intramolecular
                                   intram_mat_mdist[a_i][a_j] = std::min(intram_mat_mdist[a_i][a_j], dx2);
                                //    intram_mat_mdist[a_j][a_i] = intram_mat_mdist[a_i][a_j];
                                }
                            }
                            a_j++;
                        }
                        a_i++;
                    }
                }
                // #pragma omp parallel for num_threads(2)
                for(int ii=0; ii<natmol2[mol_id[i]]; ii++) {
                   for(int jj=ii; jj<natmol2[mol_id[i]]; jj++) {
                      if(interm_same_mat_mdist[ii][jj]<100.) {
                        kernel_density_estimator(interm_same_mat_density[mol_id[i]][ii][jj], density_bins, std::sqrt(interm_same_mat_mdist[ii][jj]), inv_num_mol[i]);
                        //interm_same_mat_density[mol_id[i]][jj][ii] = interm_same_mat_density[mol_id[i]][ii][jj];
                      } 
                      if(intram_mat_mdist[ii][jj]<100.) {
                        kernel_density_estimator(intram_mat_density[mol_id[i]][ii][jj], density_bins, std::sqrt(intram_mat_mdist[ii][jj]), inv_num_mol[i]);
                        //intram_mat_density[mol_id[i]][jj][ii] = intram_mat_density[mol_id[i]][ii][jj];
                      }
                   }
                }
                for (std::size_t j = mol_id[i]+1; j < natmol2.size(); j++) {
                   for(int ii=0; ii<natmol2[mol_id[i]]; ii++) {
                      for(int jj=0; jj<natmol2[mol_id[j]]; jj++) {
                         if(interm_cross_mat_mdist[cross_index[mol_id[i]][j]][ii][jj]<100.) {
                           kernel_density_estimator(interm_cross_mat_density[cross_index[mol_id[i]][j]][ii][jj], density_bins, std::sqrt(interm_cross_mat_mdist[cross_index[mol_id[i]][j]][ii][jj]),std::max(inv_num_mol[i],inv_num_mol[j]));
                         }
                      }
                   }
                } 
            }
            n_x++;
        }
        nframe++;
    } while (read_next_frame(oenv, status, &fr));
    close_trx(status);
    done_frame(&fr);

    sfree(xcm);
    printf("Done!\n"); fflush(stdout);

    // normalisations
    clustsize_t norm = 1.f/n_x;

    for(std::size_t i=0; i<natmol2.size(); i++) {
       for(int ii=0; ii<natmol2[i]; ii++) {
          for(int jj=ii; jj<natmol2[i]; jj++) {
                         std::transform(interm_same_mat_density[i][ii][jj].begin(), interm_same_mat_density[i][ii][jj].end(), interm_same_mat_density[i][ii][jj].begin(), [&norm](auto& c){return c*norm;});
             std::transform(intram_mat_density[i][ii][jj].begin(), intram_mat_density[i][ii][jj].end(), intram_mat_density[i][ii][jj].begin(), [&norm](auto& c){return c*norm;});

            //  std::transform(std::execution::par_unseq, interm_same_mat_density[i][ii][jj].begin(), interm_same_mat_density[i][ii][jj].end(), interm_same_mat_density[i][ii][jj].begin(), [&norm](auto& c){return c*norm;});
            //  std::transform(std::execution::par_unseq, intram_mat_density[i][ii][jj].begin(), intram_mat_density[i][ii][jj].end(), intram_mat_density[i][ii][jj].begin(), [&norm](auto& c){return c*norm;});
            interm_same_mat_density[i][jj][ii]=interm_same_mat_density[i][ii][jj];
             intram_mat_density[i][jj][ii]=intram_mat_density[i][ii][jj];          
             }
       }
       for(std::size_t j=i+1; j<natmol2.size(); j++) {
          for(int ii=0; ii<natmol2[i]; ii++) {
             for(int jj=0; jj<natmol2[j]; jj++) {
                std::transform(interm_cross_mat_density[cross_index[i][j]][ii][jj].begin(), interm_cross_mat_density[cross_index[i][j]][ii][jj].end(), interm_cross_mat_density[cross_index[i][j]][ii][jj].begin(), [&norm](auto& c){return c*norm;});
             }
          }
       }
    }

    if(write_histo) {
       for(std::size_t i=0; i<natmol2.size(); i++) {
          for(int ii=0; ii<natmol2[i]; ii++) {
             FILE *fp_inter = nullptr;
             FILE *fp_intra = nullptr;
             std::string ffh_inter = "inter_mol_"+std::to_string(i+1)+"_"+std::to_string(i+1)+"_aa_"+std::to_string(ii+1)+".dat";
             fp_inter = gmx_ffopen(ffh_inter, "w");
             std::string ffh_intra = "intra_mol_"+std::to_string(i+1)+"_"+std::to_string(i+1)+"_aa_"+std::to_string(ii+1)+".dat";
             fp_intra = gmx_ffopen(ffh_intra, "w");
             for(int k=0; k<interm_same_mat_density[i][ii][0].size(); k++) {
                fprintf(fp_inter, "%lf",  density_bins[k]);
                fprintf(fp_intra, "%lf",  density_bins[k]);
                for(int jj=0; jj<natmol2[i]; jj++) {
                      fprintf(fp_inter, " %lf",  interm_same_mat_density[i][ii][jj][k]);
                      fprintf(fp_intra, " %lf",  intram_mat_density[i][ii][jj][k]);
                }
                fprintf(fp_inter,"\n");
                fprintf(fp_intra,"\n");
             }
             gmx_ffclose(fp_inter);
             gmx_ffclose(fp_intra);
          }
          for(std::size_t j=i+1; j<natmol2.size(); j++) {
             for(int ii=0; ii<natmol2[i]; ii++) {
                FILE *fp = nullptr;
                std::string ffh = "inter_mol_"+std::to_string(i+1)+"_"+std::to_string(j+1)+"_aa_"+std::to_string(ii+1)+".dat";
                fp = gmx_ffopen(ffh, "w");
		for(int k=0; k<interm_cross_mat_density[cross_index[i][j]][ii][0].size(); k++) {
                   fprintf(fp, "%lf",  density_bins[k]);
                   for(int jj=0; jj<natmol2[j]; jj++) {
                      fprintf(fp, " %lf", interm_cross_mat_density[cross_index[i][j]][ii][jj][k]);
                   }
                   fprintf(fp, "\n");
                }
                gmx_ffclose(fp);
             }
          }
       }
    }

    for(int i=0; i<natmol2.size(); i++) {
       FILE *fp = nullptr;
       std::string inter_file_name(outfile_inter);
       std::size_t found = inter_file_name.find_last_of(".");
       fp = gmx_ffopen(inter_file_name.insert(found,"_"+std::to_string(i+1)+"_"+std::to_string(i+1)), "w");
       for(int ii=0; ii<natmol2[i]; ii++) {
          for(int jj=0; jj<natmol2[i]; jj++) {
             clustsize_t dx = cut/static_cast<clustsize_t>(interm_same_mat_density[i][ii][jj].size());
             clustsize_t dm = calc_mean(interm_same_mat_density[i][ii][jj], dx);
             clustsize_t prob = calc_prob(interm_same_mat_density[i][ii][jj], dx);
             int flag = is_flag_tolerance(interm_same_mat_density[i][ii][jj]);
             if(!flag) flag = is_slope(intram_mat_density[i][ii][jj], dx, prob);
	     clustsize_t sigma = calc_sigma(interm_same_mat_density[i][ii][jj], dx);
             clustsize_t d = is_dist(interm_same_mat_density[i][ii][jj], dx, sigma, flag);
             fprintf(fp, "%4i %4i %4i %4i %9.6lf %9.6lf %9.6lf %1i %9.6lf\n", i+1, ii+1, i+1, jj+1, dm, d, prob, flag, sigma);
          }
       }
       gmx_ffclose(fp);
       std::string intra_file_name(outfile_intra);
       found = intra_file_name.find_last_of(".");
       fp = gmx_ffopen(intra_file_name.insert(found,"_"+std::to_string(i+1)+"_"+std::to_string(i+1)), "w");
       for(int ii=0; ii<natmol2[i]; ii++) {
          for(int jj=0; jj<natmol2[i]; jj++) {
             clustsize_t dx = cut/static_cast<clustsize_t>(intram_mat_density[i][ii][jj].size());
	     clustsize_t dm = calc_mean(intram_mat_density[i][ii][jj], dx);
	     clustsize_t prob = calc_prob(intram_mat_density[i][ii][jj], dx);
             int flag = is_flag_tolerance(intram_mat_density[i][ii][jj]);
             if(!flag) flag = is_slope(intram_mat_density[i][ii][jj], dx, prob);
	     clustsize_t sigma = calc_sigma(intram_mat_density[i][ii][jj], dx);
             clustsize_t d = is_dist(intram_mat_density[i][ii][jj], dx, sigma, flag);
             fprintf(fp, "%4i %4i %4i %4i %9.6lf %9.6lf %9.6lf %1i %9.6lf\n", i+1, ii+1, i+1, jj+1, dm, d, prob, flag, sigma);
          }
       }
       gmx_ffclose(fp);
       for(int j=i+1; j<natmol2.size(); j++) {
          std::string inter_c_file_name(outfile_inter);
          found = inter_c_file_name.find_last_of(".");
          fp = gmx_ffopen(inter_c_file_name.insert(found,"_"+std::to_string(i+1)+"_"+std::to_string(j+1)), "w");
          for(int ii=0; ii<natmol2[i]; ii++) {
             for(int jj=0; jj<natmol2[j]; jj++) {
                clustsize_t dx = cut/static_cast<clustsize_t>(interm_cross_mat_density[cross_index[i][j]][ii][jj].size());
	        clustsize_t dm = calc_mean(interm_cross_mat_density[cross_index[i][j]][ii][jj], dx);
	        clustsize_t prob = calc_prob(interm_cross_mat_density[cross_index[i][j]][ii][jj], dx);
                int flag = is_flag_tolerance(interm_cross_mat_density[cross_index[i][j]][ii][jj]);
                if(!flag) flag = is_slope(interm_cross_mat_density[cross_index[i][j]][ii][jj], dx, prob);
	        clustsize_t sigma = calc_sigma(interm_cross_mat_density[cross_index[i][j]][ii][jj], dx);
                clustsize_t d = is_dist(interm_cross_mat_density[cross_index[i][j]][ii][jj], dx, sigma, flag);
                fprintf(fp, "%4i %4i %4i %4i %9.6lf %9.6lf %9.6lf %1i %9.6lf\n", i+1, ii+1, j+1, jj+1, dm, d, prob, flag, sigma);
             }
          }
          gmx_ffclose(fp);
       }
    }
}

int gmx_clustsize(int argc, char* argv[])
{
    const char* desc[] = {
        "[THISMODULE] computes the size distributions of molecular/atomic clusters in",
        "the gas phase. The output is given in the form of an [REF].xpm[ref] file.",
        "The total number of clusters is written to an [REF].xvg[ref] file.[PAR]",
        "When the [TT]-mol[tt] option is given clusters will be made out of",
        "molecules rather than atoms, which allows clustering of large molecules.",
        "In this case an index file would still contain atom numbers",
        "or your calculation will die with a SEGV.[PAR]",
        "When velocities are present in your trajectory, the temperature of",
        "the largest cluster will be printed in a separate [REF].xvg[ref] file assuming",
        "that the particles are free to move. If you are using constraints,",
        "please correct the temperature. For instance water simulated with SHAKE",
        "or SETTLE will yield a temperature that is 1.5 times too low. You can",
        "compensate for this with the [TT]-ndf[tt] option. Remember to take the removal",
        "of center of mass motion into account.[PAR]",
        "The [TT]-mc[tt] option will produce an index file containing the",
        "atom numbers of the largest cluster."
    };

    // real     kde_sig = 0.02;
    // real     cutoffino = 0.50;
    real     cutoff = 0.50;
    real     mol_cutoff = 6.00;
    real     d_pow = -6.;
    int      bOndx   = 0;
    int      nskip   = 0;
    int      skip_last_nmol = 0;
    int      nlevels = 20;
    int      ndf     = -1;
    gmx_bool bMol    = FALSE;
    gmx_bool bPBC    = TRUE;
    gmx_bool iMAT    = FALSE;
    gmx_bool iMAThis = FALSE;
    rvec     rlo     = { 1.0, 1.0, 0.0 };
    rvec     rhi     = { 0.0, 0.0, 1.0 };

    gmx_output_env_t* oenv;

    t_pargs pa[] = {
        // { "-sig",
        //   FALSE,
        //   etREAL,
        //   { &kde_sig },
        //   "Bandwidth to use for the KDE estimation" },
        // { "-cino",
        //   FALSE,
        //   etREAL,
        //   { &cutoffino },
        //   "DAJE ROMA" },
        { "-cut",
          FALSE,
          etREAL,
          { &cutoff },
          "Largest distance (nm) to be considered in a cluster" },
        { "-mol_cut",
          FALSE,
          etREAL,
          { &mol_cutoff },
          "Largest distance (nm) to be considered between molecules in a cluster" },
        { "-mol",
          FALSE,
          etBOOL,
          { &bMol },
          "Cluster molecules rather than atoms (needs [REF].tpr[ref] file)" },
        { "-inter_mol",
          FALSE,
          etBOOL,
          { &iMAT },
          "Perform an inter/intra-molecular interactions analysis (needs [REF].tpr[ref] file)" },
        { "-histo",
          FALSE,
          etBOOL,
          { &iMAThis },
          "with -inter_mol plots the histogram of the distances for all pairs (needs [REF].tpr[ref] file)" },
        { "-skip_last_nmol", FALSE, etINT, { &skip_last_nmol }, "Number of molecules to skip (from the end)" },
        { "-tr_olig_ndx",
          FALSE,
          etINT,
          { &bOndx },
          "write index files for all oligomers size from 2 to tr_olig_ndx for every frame, it could enerate A LOT of files" },
        { "-d_pow", FALSE, etREAL, { &d_pow }, "Averaging of distance^power" },
        { "-pbc", FALSE, etBOOL, { &bPBC }, "Use periodic boundary conditions" },
        { "-nskip", FALSE, etINT, { &nskip }, "Number of frames to skip between writing" },
        { "-nlevels",
          FALSE,
          etINT,
          { &nlevels },
          "Number of levels of grey in [REF].xpm[ref] output" },
        { "-ndf",
          FALSE,
          etINT,
          { &ndf },
          "Number of degrees of freedom of the entire system for temperature calculation. "
          "If not set, the number of atoms times three is used." },
        { "-rgblo",
          FALSE,
          etRVEC,
          { rlo },
          "RGB values for the color of the lowest occupied cluster size" },
        { "-rgbhi",
          FALSE,
          etRVEC,
          { rhi },
          "RGB values for the color of the highest occupied cluster size" }
    };
#define NPA asize(pa)
    const char *fnNDX, *fnTPR;
    t_rgb       rgblo, rgbhi;

    t_filenm fnm[] = {
        { efTRX, "-f", nullptr, ffREAD },         { efTPR, nullptr, nullptr, ffOPTRD },
        { efNDX, nullptr, nullptr, ffOPTRD },     { efXPM, "-o", "csize", ffWRITE },
        { efXPM, "-ow", "csizew", ffWRITE },      { efXVG, "-nc", "nclust", ffWRITE },
        { efXVG, "-mc", "maxclust", ffWRITE },    { efXVG, "-ac", "avclust", ffWRITE },
        { efXVG, "-hc", "histo-clust", ffWRITE }, { efXVG, "-temp", "temp", ffOPTWR },
        { efXVG, "-hct", "histo-time", ffWRITE },
        { efXVG, "-ict", "clust-index-time", ffWRITE },
        { efXVG, "-trm", "transitions-matrix", ffWRITE },
        { efXVG, "-km", "rates-matrix", ffWRITE },
        { efNDX, "-mcn", "maxclust", ffOPTWR },
        { efNDX, "-irmat", "intermat", ffOPTWR },
        { efNDX, "-iamat", "intramat", ffOPTWR }
    };
#define NFILE asize(fnm)

    if (!parse_common_args(
                &argc, argv, PCA_CAN_VIEW | PCA_CAN_TIME | PCA_TIME_UNIT, NFILE, fnm, NPA, pa, asize(desc), desc, 0, nullptr, &oenv))
    {
        return 0;
    }

    if(iMAT) bMol = TRUE;
 
    fnNDX   = ftp2fn_null(efNDX, NFILE, fnm);
    rgblo.r = rlo[XX];
    rgblo.g = rlo[YY];
    rgblo.b = rlo[ZZ];
    rgbhi.r = rhi[XX];
    rgbhi.g = rhi[YY];
    rgbhi.b = rhi[ZZ];

    fnTPR = ftp2fn_null(efTPR, NFILE, fnm);
    if (bMol && !fnTPR)
    {
        gmx_fatal(FARGS, "You need a tpr file for the -mol option");
    }

    if(!iMAT)
    clust_size(fnNDX,
               ftp2fn(efTRX, NFILE, fnm),
               opt2fn("-o", NFILE, fnm),
               opt2fn("-ow", NFILE, fnm),
               opt2fn("-nc", NFILE, fnm),
               opt2fn("-ac", NFILE, fnm),
               opt2fn("-mc", NFILE, fnm),
               opt2fn("-hc", NFILE, fnm),
               opt2fn("-hct", NFILE, fnm),
               opt2fn("-ict", NFILE, fnm),
               opt2fn("-trm", NFILE, fnm),
               opt2fn("-km", NFILE, fnm),
               opt2fn("-temp", NFILE, fnm),
               opt2fn("-mcn", NFILE, fnm),
               bMol,
               bPBC,
               fnTPR,
               cutoff,
               mol_cutoff,
               bOndx,
               nskip,
               skip_last_nmol,
               nlevels,
               rgblo,
               rgbhi,
               ndf,
               oenv);

    else if(iMAT)
    do_interm_mat(ftp2fn(efTRX, NFILE, fnm),
                  opt2fn("-irmat", NFILE, fnm),
                  opt2fn("-iamat", NFILE, fnm),
                //   cutoffino,
                //   kde_sig,
                  bPBC,
                  fnTPR,
                  cutoff,
                  mol_cutoff,
                  d_pow,
                  nskip,
                  skip_last_nmol,
                  iMAThis,
                  oenv);

    output_env_done(oenv);

    return 0;
}
